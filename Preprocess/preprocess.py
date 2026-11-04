"""Stage 1 preprocessing: DICOM series -> fixed-shape per-study HDF5 volumes."""
import argparse
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from tqdm import tqdm

PLANES = ("Sagittal", "Coronal", "Axial")
DICOM_EXTS = (".dcm", ".dicom", ".ima")
DEPTH, SIZE, CHANNELS = 24, 224, 3
OUT_SHAPE = (DEPTH, CHANNELS, SIZE, SIZE)
WRITE_RETRIES = 8


def list_dicoms(series_dir: Path) -> list[Path]:
    if not series_dir.is_dir():
        return []
    return [p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() in DICOM_EXTS]


def _slice_position(ds) -> float | None:
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None or len(iop) != 6 or len(ipp) != 3:
        return None
    iop = np.asarray(iop, dtype=np.float64)
    normal = np.cross(iop[:3], iop[3:])
    return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))


def load_series(files: list[Path]) -> np.ndarray | None:
    """Read, order, and rescale a series into a (D, H, W) float32 volume."""
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            px = ds.pixel_array
        except Exception:
            continue
        if px.ndim != 2:
            continue
        px = px.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        px = px * slope + intercept
        inst = getattr(ds, "InstanceNumber", None)
        slices.append({
            "px": px,
            "pos": _slice_position(ds),
            "inst": int(inst) if inst is not None else None,
            "name": f.name,
        })
    if not slices:
        return None

    # Drop stray slices whose in-plane matrix differs from the series majority.
    shape = Counter(s["px"].shape for s in slices).most_common(1)[0][0]
    slices = [s for s in slices if s["px"].shape == shape]

    if all(s["pos"] is not None for s in slices):
        slices.sort(key=lambda s: (s["pos"], s["name"]))
    elif all(s["inst"] is not None for s in slices):
        slices.sort(key=lambda s: (s["inst"], s["name"]))
    else:
        slices.sort(key=lambda s: s["name"])
    return np.stack([s["px"] for s in slices])


def normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


def resample(vol: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(vol)[None, None]
    t = F.interpolate(t, size=(DEPTH, SIZE, SIZE), mode="trilinear", align_corners=False)
    t = t[0, 0].clamp_(0.0, 1.0)
    return t[:, None].expand(-1, CHANNELS, -1, -1).contiguous().numpy()


def process_plane(dicom_dir: Path, study_uid: str, series_uids: list[str]) -> np.ndarray:
    candidates = []
    for suid in series_uids:
        files = list_dicoms(dicom_dir / study_uid / suid)
        if files:
            candidates.append((-len(files), suid, files))
    # Most files first, ties by SeriesInstanceUID; fall through if the best one is unreadable.
    for _, _, files in sorted(candidates, key=lambda c: (c[0], c[1])):
        vol = load_series(files)
        if vol is not None:
            return resample(normalize(vol))
    return np.zeros(OUT_SHAPE, dtype=np.float32)


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                all(p.lower() in f and f[p.lower()].shape == OUT_SHAPE for p in PLANES)
                and "StudyInstanceUID" in f.attrs
                and "Report" in f.attrs
            )
    except Exception:
        return False


def write_h5_atomic(path: Path, volumes: dict[str, np.ndarray], study_uid: str, report: str):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for attempt in range(WRITE_RETRIES):
        try:
            with h5py.File(tmp, "w") as f:
                for name, arr in volumes.items():
                    f.create_dataset(name, data=arr, dtype="float32", compression="lzf", chunks=(1, *OUT_SHAPE[1:]))
                f.attrs["StudyInstanceUID"] = study_uid
                f.attrs["Report"] = report
            os.replace(tmp, path)
            return
        except (OSError, BlockingIOError):
            tmp.unlink(missing_ok=True)
            if attempt == WRITE_RETRIES - 1:
                raise
            # Backoff with jitter for HDF5/NFS lock contention.
            time.sleep(min(30.0, 0.5 * 2 ** attempt) + random.random())


def process_study(task) -> tuple[str, str]:
    study_uid, report, plane_series, dicom_dir, output_dir = task
    out_path = Path(output_dir) / f"{study_uid}.h5"
    if is_complete(out_path):
        return study_uid, "cached"
    try:
        volumes = {
            p.lower(): process_plane(Path(dicom_dir), study_uid, plane_series.get(p, []))
            for p in PLANES
        }
        missing = [k for k, v in volumes.items() if not v.any()]
        write_h5_atomic(out_path, volumes, study_uid, report)
        return study_uid, f"ok (missing: {','.join(missing)})" if missing else "ok"
    except Exception as e:
        return study_uid, f"error: {type(e).__name__}: {e}"


def _init_worker():
    torch.set_num_threads(1)


def build_tasks(args) -> list[tuple]:
    split = pd.read_csv(args.train_csv, usecols=["StudyInstanceUID", "Report"])
    series = pd.read_csv(args.series_csv, usecols=["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"])
    series = series[series["Anatomical_Plane"].isin(PLANES)]
    merged = split.merge(series, on="StudyInstanceUID", how="left")

    tasks = []
    for (uid, report), grp in merged.groupby(["StudyInstanceUID", "Report"], sort=False, dropna=False):
        grp = grp.dropna(subset=["SeriesInstanceUID"])
        plane_series = grp.groupby("Anatomical_Plane")["SeriesInstanceUID"].apply(list).to_dict()
        report = "" if pd.isna(report) else str(report)
        tasks.append((str(uid), report, plane_series, args.dicom_dir, args.output_dir))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_csv", required=True, help="Split CSV with StudyInstanceUID, Report")
    ap.add_argument("--series_csv", required=True)
    ap.add_argument("--dicom_dir", required=True)
    ap.add_argument("--output_dir", required=True, help="Split-specific output dir, e.g. processed_h5/train")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)

    stats = Counter()
    errors = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for uid, status in tqdm(pool.imap_unordered(process_study, tasks, chunksize=1),
                                total=len(tasks), desc=Path(args.output_dir).name):
            stats[status.split(" ")[0].rstrip(":")] += 1
            if status.startswith("ok (missing"):
                stats["planes_zero_filled"] += 1
            if status.startswith("error"):
                errors.append(f"{uid}\t{status}")

    print(dict(stats))
    if errors:
        err_path = Path(args.output_dir) / "errors.log"
        err_path.write_text("\n".join(errors) + "\n")
        print(f"{len(errors)} studies failed; see {err_path}")


if __name__ == "__main__":
    main()
