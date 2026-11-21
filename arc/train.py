import os

import torch

from .arc import ARCSolver
from .paths import DATASET_DIR


def main():
    solver = ARCSolver(token=os.environ.get("HF_TOKEN"))
    dataset = solver.load_data(DATASET_DIR, multiply_by_300=150)
    solver.train(dataset.to_dict(orient="records"))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
