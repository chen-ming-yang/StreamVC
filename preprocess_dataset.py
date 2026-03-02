import argparse
import os
from datasets import load_dataset, Dataset, IterableDataset, Audio
import torch
from torch.utils.data import DataLoader
import soundfile as sf
import numpy as np
from einops import rearrange
import tqdm as tqdm

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_HF = "mythicinfinity/libritts"

def get_local_dataset(data_dir, split="train", streaming=True) -> Dataset | IterableDataset:
    dataset = load_dataset("audiofolder", data_dir=data_dir, split=split, streaming=streaming)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.with_format("torch")
    return dataset


def get_libritts_dataset(split, streaming=True) -> Dataset | IterableDataset:
    dataset = load_dataset(DATASET_HF, "all", split=split, streaming=streaming)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.with_format("torch")
    return dataset


class Hubert:
    def __init__(self):
        self.model = (
            torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
            .eval()
            .to(DEVICE)
        )

    @torch.no_grad()
    def get_labels(self, audio):
        single_sample_batch = rearrange(audio, "s -> 1 1 s").to(DEVICE)
        # Squeeze the first dimension if it exists and is 1, causing shape mismatch issues sometimes
        if single_sample_batch.shape[2] == 1:
             single_sample_batch = single_sample_batch.squeeze(2)
        
        labels = self.model.units(single_sample_batch).to("cpu")
        return labels


def write_audio_and_labels(id, audio, labels, save_path):
    audio_path = os.path.join(save_path, f"{id}.ogg")
    labels_path = os.path.join(save_path, f"{id}.npy")
    sf.write(audio_path, audio.detach().cpu().numpy(), SAMPLE_RATE)
    np.save(labels_path, labels.detach().cpu().numpy().astype(np.uint8))


def main(args):
    # Use split name from args if provided, otherwise default to a generic name
    split_name = os.path.basename(args.local_path) if args.local_path else args.split
    save_path = os.path.join(args.path, split_name)
    os.makedirs(save_path, exist_ok=True)
    
    if args.local_path:
        if args.verbose:
            print(f"Loading local dataset from {args.local_path} (streaming={args.streaming})")
        # Load from local directory
        dataset = get_local_dataset(args.local_path, streaming=args.streaming)
    else:
        if args.verbose:
            print(f"Loading LibriTTS split {args.split} (streaming={args.streaming})")
        dataset = get_libritts_dataset(args.split, streaming=args.streaming)
        
    if args.verbose:
        print(f"Creating DataLoader (batch_size=1, num_workers={args.num_workers})")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    if args.verbose:
        print("Loading HuBERT model (first run may download weights)...")
    hubert = Hubert()
    pbar = None
    for i, sample in enumerate(dataloader):
        if i % args.shard_length == 0:
            if pbar:
                pbar.close()
            print(f"Processing shard {i // args.shard_length}")
            pbar = tqdm.tqdm(total=args.shard_length)
        pbar.update(1)

        shard_number = i // args.shard_length
        shard_path = os.path.join(save_path, f"shard_{shard_number}")
        os.makedirs(shard_path, exist_ok=True)

        # Handle ID generation differently for local audiofolders if "id" column is missing
        if "id" in sample:
            sample_id = sample["id"][0]
        else:
            # Fallback: use filename from path or just index
            audio_path = sample["audio"]["path"][0]
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            sample_id = filename

        write_audio_and_labels(
            id=sample_id,
            audio=sample["audio"]["array"][0],
            labels=hubert.get_labels(sample["audio"]["array"][0]),
            save_path=shard_path,
        )
    if pbar:
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_path",
        type=str,
        default="/root/autodl-tmp/cmy/StreamVC/LibriSpeech/train-clean-100",
        help="Path to the local dataset folder (e.g. /path/to/train-clean-100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train.clean.100",
        choices=[
            "train.clean.100",
            "train.clean.360",
            "train.other.500",
            "dev.clean",
            "dev.other",
            "test.clean",
            "test.other",
        ],
        help="The split of the LibriTTS dataset to preprocess",
    )
    parser.add_argument(
        "--shard_length",
        type=int,
        default=5000,
        help="The number of samples to include in each directory on disk",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./dataset",
        help="The path to save the preprocessed datasets",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (set 0 if hanging)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming dataset loading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress/debug logs",
    )
    args = parser.parse_args()
    main(args)
