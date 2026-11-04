"""Stage 2: multi-view knee MRI tensors and multilingual report batches.

Each sample's ``images`` is float32 [views, slices, channels, height, width],
ordered sagittal, coronal, axial. Batches prepend a batch dimension. For a
3D encoder, use images.permute(0, 1, 3, 2, 4, 5) to put channels before depth.
Transforms receive the entire stacked sample and must preserve its shape.
CSV reports are authoritative; HDF5 attributes are not needed for text loading.
``is_valid`` marks successful image reads so training can mask fallback samples.

Example: python dataset.py --h5-dir processed_h5/train --num-workers 0
Pass --tokenizer with a Hugging Face model name or local tokenizer directory
to include tokenization and print decoded text. No tokenizer is downloaded by
default. Dependencies: torch, pandas, numpy, h5py; transformers is optional.
"""

import argparse
import logging
import time
from functools import partial
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)
PLANES = ("sagittal", "coronal", "axial")
VOLUME_SHAPE = (24, 3, 224, 224)
IMAGE_SHAPE = (len(PLANES), *VOLUME_SHAPE)


class KneeMRIDataset(Dataset):
    """Load one study per CSV row without keeping HDF5 handles across workers."""

    def __init__(
        self,
        csv_path: str | Path,
        h5_dir: str | Path,
        tokenizer=None,
        max_length: int = 512,
        transforms: Callable | None = None,
        read_retries: int = 2,
        retry_delay: float = 0.05,
    ):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if read_retries < 0 or retry_delay < 0:
            raise ValueError("read_retries and retry_delay must be nonnegative")
        self.csv_path = Path(csv_path)
        self.h5_dir = Path(h5_dir)
        self.data = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        missing = {"StudyInstanceUID", "Report"} - set(self.data.columns)
        if missing:
            raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
        for study_uid in self.data["StudyInstanceUID"]:
            if (
                not study_uid.strip()
                or study_uid in {".", ".."}
                or "/" in study_uid
                or "\\" in study_uid
                or "\x00" in study_uid
            ):
                raise ValueError(f"Invalid StudyInstanceUID: {study_uid!r}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transforms = transforms
        self.read_retries = read_retries
        self.retry_delay = retry_delay

    def __len__(self) -> int:
        return len(self.data)

    def _read_images(self, path: Path) -> tuple[torch.Tensor, bool]:
        for attempt in range(self.read_retries + 1):
            try:
                with h5py.File(path, "r") as handle:
                    images = np.empty(IMAGE_SHAPE, dtype=np.float32)
                    for plane_index, plane in enumerate(PLANES):
                        volume = handle[plane]
                        if volume.shape != VOLUME_SHAPE:
                            raise ValueError(
                                f"{plane} has shape {volume.shape}, expected {VOLUME_SHAPE}"
                            )
                        volume.read_direct(images[plane_index])
                    if not np.isfinite(images).all():
                        raise ValueError("Image contains non-finite values")
                return torch.from_numpy(images), True
            except FileNotFoundError as error:
                failure = error
                break
            except OSError as error:
                failure = error
                if attempt < self.read_retries:
                    time.sleep(self.retry_delay * 2 ** attempt)
                    continue
                break
            except (KeyError, ValueError, TypeError, RuntimeError) as error:
                failure = error
                break
        LOGGER.warning("Using zero-filled MRI fallback for %s: %s", path, failure)
        return torch.zeros(IMAGE_SHAPE, dtype=torch.float32), False

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        study_uid, report = row["StudyInstanceUID"], row["Report"]
        images, is_valid = self._read_images(self.h5_dir / f"{study_uid}.h5")
        if self.transforms is not None and is_valid:
            images = self.transforms(images)
            if not isinstance(images, torch.Tensor) or tuple(images.shape) != IMAGE_SHAPE:
                raise ValueError(f"transforms must return a tensor with shape {IMAGE_SHAPE}")
            images = images.to(dtype=torch.float32)
        sample = {
            "images": images,
            "StudyInstanceUID": study_uid,
            "report": report,
            "is_valid": is_valid,
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                report,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )
            for key in ("input_ids", "attention_mask"):
                sample[key] = torch.as_tensor(encoded[key], dtype=torch.long)
        return sample


def collate_knee_mri(samples: list[dict], tokenizer=None) -> dict:
    """Stack images and dynamically pad text using the tokenizer's pad policy."""
    batch = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "StudyInstanceUID": [sample["StudyInstanceUID"] for sample in samples],
        "report": [sample["report"] for sample in samples],
        "is_valid": torch.tensor([sample["is_valid"] for sample in samples], dtype=torch.bool),
    }
    if tokenizer is not None:
        encoded = [
            {key: sample[key].tolist() for key in ("input_ids", "attention_mask")}
            for sample in samples
        ]
        padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        batch.update({key: padded[key] for key in ("input_ids", "attention_mask")})
    return batch


def get_dataloader(
    csv_path: str | Path,
    h5_dir: str | Path,
    tokenizer=None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    *,
    max_length: int = 512,
    transforms: Callable | None = None,
    pin_memory: bool | None = None,
    read_retries: int = 2,
    retry_delay: float = 0.05,
) -> DataLoader:
    """Build a loader; tokenizer must define a pad token when supplied.

    Each sample occupies about 41 MiB before batching. Worker prefetch is
    limited to one batch per worker to reduce host/shared-memory pressure.
    With multiprocessing spawn, tokenizers and transforms must be picklable.
    """
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("The tokenizer needs a pad_token_id for batch padding")
    dataset = KneeMRIDataset(
        csv_path, h5_dir, tokenizer, max_length, transforms, read_retries, retry_delay
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_knee_mri, tokenizer=tokenizer),
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        **({"prefetch_factor": 1} if num_workers > 0 else {}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=Path(__file__).parent / "Split/train.csv")
    parser.add_argument("--h5-dir", type=Path, default=Path(__file__).parent / "processed_h5")
    parser.add_argument("--tokenizer", help="Hugging Face model name or local tokenizer directory")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    loader = get_dataloader(
        args.csv_path,
        args.h5_dir,
        tokenizer,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )
    if not len(loader.dataset):
        parser.error("The split CSV contains no studies")
    batch = next(iter(loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"StudyInstanceUID: {batch['StudyInstanceUID'][0]}")
    print(f"Valid image: {batch['is_valid'][0].item()}")
    print(f"Report: {batch['report'][0]}")
    if tokenizer is not None:
        print(f"Decoded report: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()