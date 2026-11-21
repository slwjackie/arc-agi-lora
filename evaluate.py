import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arc.paths import DATASET_DIR


def check_match(pred, truth):
    try:
        pred = np.asarray(pred)
    except (TypeError, ValueError):
        return 0
    truth = np.asarray(truth)

    print(pred)
    print(truth)
    print()
    return int(pred.ndim == 2 and np.array_equal(pred, truth))


def load_data(base_dir, sample_count=1000):
    tasks = []
    for path in sorted(Path(base_dir).glob("*.json")):
        if path.is_file():
            with path.open(encoding="utf-8") as fp:
                tasks.append((path.stem, json.load(fp)))
    if not tasks:
        raise ValueError(f"No task JSON files found in {base_dir}")

    data = []
    rng = np.random.default_rng(42)
    for _ in range(sample_count):
        name, task = tasks[rng.integers(len(tasks))]
        indices = rng.choice(len(task), size=4, replace=True)
        train = [task[i] for i in indices[:3]]
        test = task[indices[3]]
        data.append({
            "task": name,
            "train": train,
            "test_input": [{"input": test["input"]}],
            "test_output": [test["output"]],
            "test": [{"input": test["input"], "output": test["output"]}],
        })
    return pd.DataFrame(data)


def main():
    import torch
    from datasets import Dataset
    from tqdm.auto import tqdm
    from transformers import set_seed

    from arc import ARCSolver

    torch.cuda.empty_cache()
    solver = ARCSolver(token=os.environ.get("HF_TOKEN"))
    solver.prepare_evaluation()
    set_seed(1234567890)

    df = load_data(DATASET_DIR)
    eval_dataset = Dataset.from_pandas(df).shuffle(42).select(range(10))
    scores = []
    for item in tqdm(eval_dataset):
        test = item["test"][0]
        prediction = solver.predict(item["train"], test["input"])
        scores.append(check_match(prediction, test["output"]))

    print(f"Evaluation scores: {np.mean(scores) * 100:.2f}", flush=True)
    print("Evaluation Success")


if __name__ == "__main__":
    main()
