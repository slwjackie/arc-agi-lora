from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ADAPTER_DIR = ARTIFACTS_DIR / "checkpoint-qwen-4B-150"
