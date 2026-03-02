import argparse
import glob
import os
from typing import Iterable

import safetensors as st
import safetensors.torch
import soundfile as sf
import torch
import torchaudio.functional as F

from streamvc import StreamVC

SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _load_raw_checkpoint(path: str) -> dict:
    if path.endswith(".safetensors"):
        return st.torch.load_file(path, device=DEVICE)
    return torch.load(path, map_location=DEVICE)


def _unwrap_state_dict(raw_checkpoint: dict) -> dict:
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Checkpoint object must be dict, got {type(raw_checkpoint)}")

    for key in ("state_dict", "model_state_dict", "model"):
        maybe_state_dict = raw_checkpoint.get(key)
        if isinstance(maybe_state_dict, dict):
            return maybe_state_dict
    return raw_checkpoint


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _candidate_checkpoint_paths(checkpoint_path: str) -> list[str]:
    if os.path.isdir(checkpoint_path):
        candidates = sorted(glob.glob(os.path.join(checkpoint_path, "pytorch_model*.bin")))
        candidates += sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
        return candidates

    candidates = [checkpoint_path]
    base = os.path.basename(checkpoint_path)
    if base.startswith("pytorch_model") and checkpoint_path.endswith(".bin"):
        parent = os.path.dirname(checkpoint_path) or "."
        for path in sorted(glob.glob(os.path.join(parent, "pytorch_model*.bin"))):
            if path not in candidates:
                candidates.append(path)
    return candidates


def _normalized_state_dict_variants(state_dict: dict) -> Iterable[dict]:
    yield state_dict
    module_stripped = _strip_prefix_if_present(state_dict, "module.")
    yield module_stripped
    model_stripped = _strip_prefix_if_present(module_stripped, "model.")
    yield model_stripped


def _select_best_state_dict(model: StreamVC, checkpoint_path: str) -> tuple[dict, str]:
    model_keys = set(model.state_dict().keys())
    candidates = _candidate_checkpoint_paths(checkpoint_path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found from path: {checkpoint_path}")

    best_state_dict = None
    best_path = None
    best_match_count = -1

    for path in candidates:
        raw = _load_raw_checkpoint(path)
        state_dict = _unwrap_state_dict(raw)

        for variant in _normalized_state_dict_variants(state_dict):
            match_count = sum(1 for key in variant.keys() if key in model_keys)
            if match_count > best_match_count:
                best_match_count = match_count
                best_state_dict = variant
                best_path = path

    if best_state_dict is None or best_match_count <= 0:
        raise RuntimeError(
            f"Could not find StreamVC-compatible checkpoint in {checkpoint_path}. "
            "If you pass an accelerate state folder, make sure it contains pytorch_model*.bin files."
        )

    return best_state_dict, best_path


@torch.no_grad()
def main(args):
    """Main function for StreamVC model inference."""
    model = StreamVC().to(device=DEVICE, dtype=DTYPE).eval()
    state_dict, loaded_from = _select_best_state_dict(model, args.checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint loaded from '{loaded_from}' is not compatible with StreamVC. "
            f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}; "
            f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}."
        )
    print(f"Loaded checkpoint from: {loaded_from}")

    source_speech, orig_sr = sf.read(args.source_speech)
    # Ensure mono 1D audio (model expects shape [..., time])
    if source_speech.ndim > 1:
        source_speech = source_speech.mean(axis=-1)
    source_speech = torch.from_numpy(source_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        source_speech = F.resample(source_speech, orig_sr, SAMPLE_RATE)

    target_speech, orig_sr = sf.read(args.target_speech)
    # Ensure mono 1D audio (model expects shape [..., time])
    if target_speech.ndim > 1:
        target_speech = target_speech.mean(axis=-1)
    target_speech = torch.from_numpy(target_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        target_speech = F.resample(target_speech, orig_sr, SAMPLE_RATE)
    # source_speech = source_speech / (torch.abs(source_speech).max() + 1e-6)
    # target_speech = target_speech / (torch.abs(target_speech).max() + 1e-6)
    output = model(source_speech, target_speech)
    output = output.cpu().squeeze()
    # Prevent clipping / extreme amplitudes in output
    output = torch.clamp(output, -1.0, 1.0).numpy()
    sf.write(args.output_path, output, SAMPLE_RATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='StreamVC Inference Script',
        description='Inference script for StreamVC model, performs voice conversion on a single audio file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--checkpoint", type=str,
                        default="./model.safetensors",
                        help="A path the a pretrained StreamVC model checkpoint (safetensors).")
    parser.add_argument("-s", "--source-speech", type=str,
                        help="A path to a an audio file with the source speech input for the model.")
    parser.add_argument("-t", "--target-speech", type=str,
                        help="A path to a an audio file with the target speech input for the model.")
    parser.add_argument("-o", "--output-path", type=str,
                        default="./out.wav",
                        help="Output file path.")

    main(parser.parse_args())
