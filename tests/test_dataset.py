import logging

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from dataset import IMAGE_SHAPE, PLANES, VOLUME_SHAPE, KneeMRIDataset, get_dataloader


class DummyTokenizer:
    pad_token_id = 7

    def __call__(self, text, *, truncation, max_length, padding, return_attention_mask):
        tokens = [ord(character) for character in text][:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def pad(self, encoded, *, padding, return_tensors):
        length = max(len(item["input_ids"]) for item in encoded)
        return {
            key: torch.tensor([
                item[key] + [fill] * (length - len(item[key])) for item in encoded
            ], dtype=torch.long)
            for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0))
        }


def make_split(tmp_path, reports=("normal",)):
    path = tmp_path / "split.csv"
    pd.DataFrame({
        "StudyInstanceUID": [f"001.{index}" for index in range(len(reports))],
        "Report": reports,
    }).to_csv(path, index=False)
    return path


def make_h5(tmp_path, uid="001.0", invalid=None):
    with h5py.File(tmp_path / f"{uid}.h5", "w") as handle:
        for plane_index, plane in enumerate(PLANES):
            if invalid == "missing_plane" and plane == "axial":
                continue
            shape = (1,) if invalid == "shape" else VOLUME_SHAPE
            value = np.nan if invalid == "nonfinite" else (plane_index + 1) / 4
            handle.create_dataset(plane, shape=shape, dtype="float32", fillvalue=value)


def test_loads_view_order_dtype_and_transform(tmp_path):
    csv_path = make_split(tmp_path)
    make_h5(tmp_path)
    sample = KneeMRIDataset(csv_path, tmp_path, transforms=lambda images: images + 1)[0]
    assert sample["images"].shape == IMAGE_SHAPE
    assert sample["images"].dtype == torch.float32
    assert sample["StudyInstanceUID"] == "001.0"
    assert sample["is_valid"]
    assert sample["images"][:, 0, 0, 0, 0].tolist() == [1.25, 1.5, 1.75]


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "missing_plane", "shape", "nonfinite"])
def test_bad_files_return_zero_fallback(tmp_path, caplog, invalid):
    csv_path = make_split(tmp_path)
    if invalid == "corrupt":
        (tmp_path / "001.0.h5").write_bytes(b"not an hdf5 file")
    elif invalid != "missing":
        make_h5(tmp_path, invalid=invalid)
    with caplog.at_level(logging.WARNING):
        sample = KneeMRIDataset(csv_path, tmp_path, read_retries=0)[0]
    assert not sample["is_valid"]
    assert sample["images"].shape == IMAGE_SHAPE
    assert torch.count_nonzero(sample["images"]) == 0
    assert sample["report"] == "normal"
    assert "001.0.h5" in caplog.text


def test_unicode_truncation_padding_and_empty_reports(tmp_path):
    reports = ("\u819d\u5173\u8282\u6b63\u5e38", "NA", "")
    csv_path = make_split(tmp_path, reports)
    loader = get_dataloader(
        csv_path, tmp_path, DummyTokenizer(), 3, shuffle=False,
        num_workers=0, max_length=3, read_retries=0,
    )
    batch = next(iter(loader))
    assert batch["images"].shape == (3, *IMAGE_SHAPE)
    assert batch["report"] == list(reports)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].tolist() == [[33181, 20851, 33410], [78, 65, 7], [7, 7, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0], [0, 0, 0]]
    assert batch["is_valid"].tolist() == [False, False, False]


def test_worker_loading_without_tokenizer(tmp_path):
    csv_path = make_split(tmp_path)
    make_h5(tmp_path)
    loader = get_dataloader(csv_path, tmp_path, batch_size=1, shuffle=False, num_workers=1)
    batch = next(iter(loader))
    assert batch["images"].shape == (1, *IMAGE_SHAPE)
    assert batch["is_valid"].tolist() == [True]
    assert "input_ids" not in batch


def test_retries_transient_open_error(tmp_path, monkeypatch):
    csv_path = make_split(tmp_path)
    make_h5(tmp_path)
    original_open = h5py.File
    attempts = []

    def flaky_open(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise BlockingIOError("file is locked")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", flaky_open)
    assert KneeMRIDataset(csv_path, tmp_path, retry_delay=0)[0]["is_valid"]
    assert len(attempts) == 2


def test_rejects_bad_metadata_and_missing_padding_token(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"Report": ["normal"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        KneeMRIDataset(path, tmp_path)
    pd.DataFrame({"StudyInstanceUID": ["../escape"], "Report": [""]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid StudyInstanceUID"):
        KneeMRIDataset(path, tmp_path)
    tokenizer = DummyTokenizer()
    tokenizer.pad_token_id = None
    with pytest.raises(ValueError, match="pad_token_id"):
        get_dataloader(make_split(tmp_path), tmp_path, tokenizer)