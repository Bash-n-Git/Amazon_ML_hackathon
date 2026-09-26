"""Paths and global settings. Override the data root with the BER_DATA env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # .../Amazon ML
DATA = Path(os.environ.get("BER_DATA", ROOT / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUT = Path(os.environ.get("BER_OUT", ROOT / "output"))
SEED = 42

WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Keep every temp/cache file next to the project (nothing on the system drive).
CACHE = Path(os.environ.get("BER_CACHE", ROOT / ".cache"))
for var, sub in {"TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp", "JOBLIB_TEMP_FOLDER": "tmp",
                 "POLARS_TEMP_DIR": "tmp", "HF_HOME": "hf", "TORCH_HOME": "torch",
                 "CUDA_CACHE_PATH": "cuda", "PIP_CACHE_DIR": "pip"}.items():
    (CACHE / sub).mkdir(parents=True, exist_ok=True)
    os.environ[var] = str(CACHE / sub)
import tempfile  # noqa: E402
tempfile.tempdir = str(CACHE / "tmp")


def src_tsv(split: str, i: int) -> Path:
    return DATA / split / f"{split}_source{i}.tsv"


def pq(name: str) -> Path:
    return WORK / f"{name}.parquet"
