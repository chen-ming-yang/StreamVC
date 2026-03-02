import abc
from dataclasses import dataclass
import os
from itertools import islice
from typing import Union

import accelerate.checkpointing
import safetensors as st
import safetensors.torch
import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from streamvc.model import StreamVC
from streamvc.train.discriminator import Discriminator
from streamvc.train.encoder_classifier import EncoderClassifier
from streamvc.train.data import PreprocessedDataset
from streamvc.train.loss import (
    GeneratorLoss,
    DiscriminatorLoss,
    FeatureLoss,
    ReconstructionLoss,
)
import tyro
import config.lr_scheduler as scheduler_config
from config.training_config import UnifiedTrainingConfig
from config.utils import get_flattened_config_dict
from typing_extensions import Annotated
import logging

logger = get_logger(__name__)

accelerator = Accelerator(
    log_with="tensorboard",
    project_config=ProjectConfiguration(
        project_dir=os.getcwd(), logging_dir=os.path.join(os.getcwd(), "logs")
    ),
    dataloader_config=DataLoaderConfiguration(split_batches=True),
)

NUM_CLASSES = 100
EMBEDDING_DIMS = 64
SAMPLES_PER_FRAME = 320
DEVICE = accelerator.device

####### cli commands #######


####### tensorboard logging functions #######


@accelerator.on_main_process
def log_gradients_tensorboard(model, step, prefix=""):
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    for name, param in model.named_parameters():
        if param.grad is not None:
            summary_writer.add_histogram(
                f"gradients/{[prefix]}{name}", param.grad, global_step=step
            )


@accelerator.on_main_process
def log_labels_tensorboard(outputs_flat, labels_flat, step):
    _, predicted = torch.max(outputs_flat.data, 1)
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    summary_writer.add_histogram("labels/content_encoder", predicted, global_step=step)
    summary_writer.add_histogram("labels/hubert", labels_flat, global_step=step)


####### LR schedulers #######


def get_lr_Scheduler(
    optimizer: torch.optim.Optimizer,
    config: UnifiedTrainingConfig,
    total_steps: int,
    discriminator: bool = False,
):
    scheduler = config.lr_scheduler
    if scheduler is None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif isinstance(scheduler, scheduler_config.StepLR):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler.step, gamma=scheduler.gamma
        )
    elif isinstance(scheduler, scheduler_config.LinearLR):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler.start,
            end_factor=scheduler.end,
            total_iters=total_steps,
        )
    elif isinstance(scheduler, scheduler_config.ExponentialLR):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler.gamma)
    elif isinstance(scheduler, scheduler_config.OneCycleLR):
        max_lr = scheduler.max
        if discriminator:
            max_lr = config.lr_discriminator_multiplier * max_lr
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=scheduler.pct_start,
            div_factor=scheduler.div_factor,
            final_div_factor=scheduler.final_div_factor,
        )
    elif isinstance(scheduler, scheduler_config.CosineAnnealingWarmRestarts):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler.T_0,
            T_mult=1,
            eta_min=scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


####### Savable counter #######


@dataclass
class CounterState:
    value: int = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(self.value, dtype=torch.long, device="cpu")}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.value = state_dict["value"].item()


####### Training #######


class TrainerBase(abc.ABC):
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        config: UnifiedTrainingConfig = self.config
        self.train_dataset = PreprocessedDataset(config.datasets.train_dataset_path)
        self.train_dataloader = self.train_dataset.get_dataloader(
            config.batch_size, limit_samples=config.limit_batch_samples
        )

        self.steps_per_epoch = max(
            [
                i
                for i in [len(self.train_dataloader), config.limit_num_batches]
                if i is not None
            ]
        )
        self.total_steps = self.steps_per_epoch * config.num_epochs

        self.dev_dataset = PreprocessedDataset(config.datasets.dev_dataset_path)
        self.dev_dataloader = self.dev_dataset.get_dataloader(
            config.batch_size, limit_samples=config.limit_batch_samples
        )
        self.models = {}
        self._prepered_training = False

    @abc.abstractmethod
    def prepare_training(self) -> None:
        assert not self._prepered_training, "prepare_training can only be called once"
        self._prepered_training = True

    @abc.abstractmethod
    def train_step(
        self,
        batch: torch.Tensor,
        lables: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]: ...

    def save_state(self, epoch: int | None = None):
        """Save training state checkpoint.
        
        Args:
            epoch: If provided, save as versioned epoch checkpoint.
                   If None, save as latest checkpoint (overwriting).
        """
        accelerator.wait_for_everyone()
        
        if epoch is not None:
            # Versioned checkpoint by epoch
            checkpoint_dir = os.path.join(
                self.config.model_checkpoint_path, 
                f"{self.config.run_name}_state_epoch{epoch}"
            )
        else:
            # Latest checkpoint (for step-based saves)
            checkpoint_dir = os.path.join(
                self.config.model_checkpoint_path, f"{self.config.run_name}_state"
            )
        
        accelerator.save_state(checkpoint_dir, safe_serialization=False)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Clean up old checkpoints if max_checkpoints is set
        if epoch is not None and self.config.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

    def save_models(self):
        for name, model in self.models.items():
            accelerator.save_model(
                model,
                save_directory=os.path.join(
                    self.config.model_checkpoint_path,
                    f"{self.config.run_name}_{name}",
                ),
                safe_serialization=False,
            )
    
    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the most recent ones."""
        import glob
        import re
        
        pattern = os.path.join(
            self.config.model_checkpoint_path,
            f"{self.config.run_name}_state_epoch*"
        )
        checkpoint_dirs = sorted(glob.glob(pattern))
        
        # Extract epoch numbers and sort
        epoch_checkpoints = []
        for ckpt_dir in checkpoint_dirs:
            match = re.search(r'_epoch(\d+)$', ckpt_dir)
            if match:
                epoch_checkpoints.append((int(match.group(1)), ckpt_dir))
        
        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints beyond max_checkpoints
        if len(epoch_checkpoints) > self.config.max_checkpoints:
            for _, old_dir in epoch_checkpoints[self.config.max_checkpoints:]:
                import shutil
                try:
                    shutil.rmtree(old_dir)
                    logger.info(f"Removed old checkpoint: {old_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_dir}: {e}")

    def train(self) -> None:
        config = self.config
        if not self._prepered_training:
            self.prepare_training()

        if not hasattr(self, "global_step"):
            self.global_step = CounterState()
            accelerator.register_for_checkpointing(self.global_step)
        else:
            self.global_step.value = 0

        if config.restore_state_dir is not None:
            accelerator.load_state(config.restore_state_dir)

        start_epoch = self.global_step.value // self.steps_per_epoch
        start_step = self.global_step.value % self.steps_per_epoch

        from collections import defaultdict
        losses_aggregate = defaultdict(list)
        for epoch in range(start_epoch, config.num_epochs):
            logger.info(f"epoch num: {epoch}")

            dataloader = self.train_dataloader
            if start_step != 0:
                dataloader = accelerator.skip_first_batches(
                    self.train_dataloader, start_step
                )

            for step, (batch, labels, mask) in enumerate(
                islice(dataloader, config.limit_num_batches),
                start=start_step,
            ):
                losses = self.train_step(batch, labels, mask)

                if isinstance(losses, dict):
                    for k, v in losses.items():
                        losses_aggregate[k].append(v)
                else: # fallback
                    losses_aggregate["loss"].append(losses)
                
                if (self.global_step.value + 1) % config.print_interval == 0:
                    loss_str = ", ".join(f"{k}: {torch.tensor(v).mean().item():.4f}" for k, v in losses_aggregate.items())
                    logger.info(
                        f"[{epoch}, {step:5}] {loss_str}"
                    )
                    losses_aggregate.clear()

                # Step-based checkpoint (overwrites)
                if config.model_checkpoint_interval > 0 and (self.global_step.value + 1) % config.model_checkpoint_interval == 0:
                    self.save_state()

                self.after_train_step()
                self.global_step.value += 1
            start_step = 0
            
            # Epoch-based checkpoint (versioned)
            if config.epoch_checkpoint_interval > 0 and (epoch + 1) % config.epoch_checkpoint_interval == 0:
                self.save_state(epoch=epoch + 1)

        self.save_state(epoch=config.num_epochs)
        self.save_models()

    def after_train_step(self) -> None:
        pass


class UnifiedTrainer(TrainerBase):
    def __init__(self, config: UnifiedTrainingConfig):
        super().__init__(config)
        streamvc = StreamVC(gradient_checkpointing=config.gradient_checkpointing)
        content_encoder = streamvc.content_encoder
        self.wrapped_content_encoder = EncoderClassifier(
            content_encoder, EMBEDDING_DIMS, NUM_CLASSES, dropout=config.dropout
        )
        self.generator = streamvc
        self.discriminator = Discriminator(
            gradient_checkpointing=config.gradient_checkpointing
        )
        self.models["generator"] = self.generator
        self.models["discriminator"] = self.discriminator
        self.models["content_encoder_classifier"] = self.wrapped_content_encoder
        

    def prepare_training(self) -> None:
        super().prepare_training()

        config = self.config
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        optimizer_content_encoder = torch.optim.AdamW(
            params=self.wrapped_content_encoder.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        content_encoder_param_ids = {
            id(p) for p in self.generator.content_encoder.parameters()
        }

        optimizer_generator = torch.optim.AdamW(
            params=[
                param
                for param in self.generator.parameters()
                if id(param) not in content_encoder_param_ids
            ],
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )


        lr_discriminator = config.lr_discriminator_multiplier * config.lr
        optimizer_discriminator = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=lr_discriminator,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        scheduler_content_encoder = get_lr_Scheduler(
            optimizer_content_encoder, config, self.total_steps
        )
        scheduler_generator = get_lr_Scheduler(
            optimizer_generator, config, self.total_steps
        )
        scheduler_discriminator = get_lr_Scheduler(
            optimizer_discriminator, config, self.total_steps, discriminator=True
        )

        generator_loss_fn = GeneratorLoss()
        discriminator_loss_fn = DiscriminatorLoss()
        feature_loss_fn = FeatureLoss()
        reconstruction_loss_fn = ReconstructionLoss(
            gradient_checkpointing=config.gradient_checkpointing
        )

        [
            self.wrapped_content_encoder,
            self.generator,
            self.discriminator,
            self.optimizer_content_encoder,
            self.optimizer_generator,
            self.optimizer_discriminator,
            self.scheduler_content_encoder,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.train_dataloader,
            self.criterion,
            self.generator_loss_fn,
            self.discriminator_loss_fn,
            self.feature_loss_fn,
            self.reconstruction_loss_fn,
        ] = accelerator.prepare(
            self.wrapped_content_encoder,
            self.generator,
            self.discriminator,
            optimizer_content_encoder,
            optimizer_generator,
            optimizer_discriminator,
            scheduler_content_encoder,
            scheduler_generator,
            scheduler_discriminator,
            self.train_dataloader,
            self.criterion,
            generator_loss_fn,
            discriminator_loss_fn,
            feature_loss_fn,
            reconstruction_loss_fn,
        )


    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Annotated[float, "loss"]:


        config = self.config
        step = self.global_step.value
        log_grads = config.log_gradient_interval and (step + 1) % config.log_gradient_interval == 0


        #phase 1: content encoder classification loss
        self.optimizer_content_encoder.zero_grad()

        ce_outputs = self.wrapped_content_encoder(batch)
        ce_outputs_flat = ce_outputs.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)
        ce_loss = self.criterion(ce_outputs_flat, labels_flat)

        accelerator.backward(config.lambda_conetent_encoder * ce_loss)

        if hasattr(config, "max_grad_norm") and config.max_grad_norm is not None:
            accelerator.clip_grad_norm_(
                self.wrapped_content_encoder.parameters(), max_norm=config.max_grad_norm
            )
        
        self.optimizer_content_encoder.step()
        self.scheduler_content_encoder.step(step)


        if (
            self.config.log_gradient_interval
            and (self.global_step.value + 1) % self.config.log_gradient_interval == 0
        ):
            log_gradients_tensorboard(self.wrapped_content_encoder, self.global_step.value)
        
        if log_grads:
            log_gradients_tensorboard(self.wrapped_content_encoder, self.global_step.value, prefix="content_encoder_")


        # phase 2: generator update
        self.optimizer_generator.zero_grad()
        x_pred_t = self.generator(batch, batch)
        # Remove the first 2 frames from the generated audio
        # because we match a output frame t with input frame t-2.
        x_pred_t = x_pred_t[..., SAMPLES_PER_FRAME * 2 :]
        batch_trimmed = batch[..., : x_pred_t.shape[-1]]
        mask_ratio = mask.sum(dim=-1) / mask.shape[-1]


        discriminator_fake = self.discriminator(x_pred_t)

        with torch.no_grad():
            discriminator_real_for_further = self.discriminator(batch_trimmed)
        
        adversarial_loss = self.generator_loss_fn(discriminator_fake, mask_ratio)
        feature_loss = self.feature_loss_fn(
            discriminator_real_for_further, discriminator_fake, mask_ratio
        )
        reconstrunction_loss = self.reconstruction_loss_fn(batch_trimmed, x_pred_t, mask_ratio)

        generator_loss = (config.lambda_adversarial * adversarial_loss 
                          + config.lambda_feature * feature_loss 
                          + config.lambda_reconstruction * reconstrunction_loss)

        accelerator.backward(
            generator_loss
        )

        if hasattr(config, "max_grad_norm") and config.max_grad_norm is not None:
            content_encoder_param_ids = {
                id(p) for p in self.generator.content_encoder.parameters()
            }
            generator_params = [
                param
                for param in self.generator.parameters()
                if id(param) not in content_encoder_param_ids
            ]
            accelerator.clip_grad_norm_(
                generator_params, max_norm=config.max_grad_norm
            )
        
        self.optimizer_generator.step()
        self.scheduler_generator.step(step)

        if log_grads:
            log_gradients_tensorboard(self.generator, self.global_step.value, prefix="generator_")
        
        #phase 3: discriminator update
        self.optimizer_discriminator.zero_grad()

        discriminator_real = self.discriminator(batch_trimmed)
        discriminator_fake_detached = self.discriminator(x_pred_t.detach())

        discriminator_loss = self.discriminator_loss_fn(
            discriminator_real, discriminator_fake_detached, mask_ratio
        )

        accelerator.backward(discriminator_loss)


        if hasattr(config, "max_grad_norm") and config.max_grad_norm is not None:
            accelerator.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=config.max_grad_norm
            )
        
        self.optimizer_discriminator.step()
        self.scheduler_discriminator.step(step)

        if log_grads:
            log_gradients_tensorboard(self.discriminator, self.global_step.value, prefix="discriminator_")

        accelerator.log(
            {
                "loss/content_encoder": ce_loss.item(),
                "loss/generator": generator_loss.item(),
                "loss/discriminator": discriminator_loss.item(),
                "loss/adversarial": adversarial_loss.item(),
                "loss/feature_matching": feature_loss.item(),
                "loss/reconstruction": reconstrunction_loss.item(),
                "lr/content_encoder": self.scheduler_content_encoder.get_last_lr()[0],
                "lr/generator": self.scheduler_generator.get_last_lr()[0],
                "lr/discriminator": self.scheduler_discriminator.get_last_lr()[0],
            },
            step=self.global_step.value,
        )
        return {
            "ce": ce_loss.item(),
            "gen": generator_loss.item(),
            "disc": discriminator_loss.item(),
            "adv": adversarial_loss.item(),
            "feat": feature_loss.item(),
            "recon": reconstrunction_loss.item()
        }

    def after_train_step(self) -> None:
        if (self.global_step.value + 1) % self.config.accuracy_interval == 0:
            accuracy = self.compute_content_encoder_accuracy()
            accuracies = accelerator.gather_for_metrics([accuracy])
            accuracies = torch.tensor(accuracies)
            gathered_accuracy = accuracies.mean().item()
            accelerator.log(
                {"accuracy/content_encoder": gathered_accuracy},
                step=self.global_step.value,
            )
            logger.info(f"accuracy: {accuracy:.2f}%")

    @torch.no_grad()
    def compute_content_encoder_accuracy(self):
        correct = 0
        total = 0
        self.wrapped_content_encoder.eval()
        for batch, labels, _ in islice(
            self.dev_dataloader, self.config.accuracy_limit_num_batches
        ):
            batch = batch.to(accelerator.device)
            labels = labels.to(accelerator.device)
            outputs = self.wrapped_content_encoder(batch)
            outputs_flat = outputs.view(-1, NUM_CLASSES)
            labels_flat = labels.view(-1)
            _, predicted = torch.max(outputs_flat.data, 1)
            total += torch.sum(labels_flat != -1).item()
            correct += (predicted == labels_flat).sum().item()
        self.wrapped_content_encoder.train()

        return 100 * correct / total



def main(config: UnifiedTrainingConfig) -> None:
    """Main function for training StreamVC model."""
    logger.debug(f"DEVICE={accelerator.device}")
    hps = get_flattened_config_dict(config)
    hps["num processes"] = accelerator.num_processes
    hps["mixed precision"] = accelerator.mixed_precision

    if accelerator.gradient_accumulation_steps > 1:
        raise ValueError(
            "Gradient accumulation is not supported. Disable gradient accumulation from accelerate"
        )
    logger.debug(f"{hps=}")

    accelerator.init_trackers(config.run_name, config=hps)

    trainer = UnifiedTrainer(config)

    trainer.train()

    accelerator.end_training()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)
    training_config = tyro.cli(UnifiedTrainingConfig)

    # Setup file logging
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join("logs", f"{training_config.run_name}.log")
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    main(training_config)
