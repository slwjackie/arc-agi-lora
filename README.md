# ARC-AGI LoRA

This project fine-tunes Qwen3-4B to solve grid-transformation puzzles from the Abstraction and Reasoning Corpus (ARC). Each puzzle gives the model three input–output examples and a fourth input. The model must infer the rule shared by the examples and produce the entire output grid, including its dimensions. Each digit from 0 to 9 represents a color.

For example, task `67a3c6ac` asks the solver to mirror a grid from left to right. The rows stay in place, and the cells in each row appear in reverse order. Here is one input and its expected output from the [task data](dataset/67a3c6ac.json).

```text
Input    Expected output
611      116
481      184
193      391
```

## How Qwen3-4B learns the task

The training set contains 300 task files with 269,312 input–output pairs. For each task, the training script samples four distinct pairs, uses the first three as demonstrations, and uses the fourth output as the answer to predict. It repeats this 150 times per task to build 45,000 training examples. Grids are serialized as rows of digits. Each output starts with a `height*width` line, such as `3*3` for the example above.

The solver loads `Qwen/Qwen3-4B` with bitsandbytes 4-bit NF4 quantization, double quantization, and FP16 computation. It freezes the base weights and trains low-rank adapters using PEFT. [Solver implementation](arc/arc.py)

| Training setting | Value |
| --- | --- |
| LoRA rank / alpha / dropout | 8 / 64 / 0.05 |
| Attention projections with adapters | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Feed-forward projections with adapters | `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | Paged 8-bit AdamW |
| Training schedule | 2 epochs, 1 example per device per step, 8 gradient accumulation steps |

## Setup and training

Use a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation. The first training run downloads the base model from Hugging Face. From the repository root, install the dependencies and start training.

```bash
python -m pip install -r requirements.txt
python -m arc.train
```

Training saves the final adapter in `artifacts/checkpoint-qwen-4B-150/` and writes a checkpoint every 1,000 optimizer steps. Rerunning the training command resumes from the latest available checkpoint. To inspect a sample puzzle before training, run `python examples.py`.

## Evaluation

After training, run the evaluator from the repository root.

```bash
python evaluate.py
```

The evaluator loads the saved adapter onto Qwen3-4B. It builds a seeded pool of 1,000 puzzles from the same 300 task files used for training, shuffles the pool, and scores 10 puzzles. It samples pairs within each puzzle with replacement, so a query can repeat a demonstration.

The script prints each prediction alongside its target and reports the percentage of exact matches. A prediction counts as correct only when its dimensions and every cell match the target. The score measures performance on these sampled puzzles from familiar task families. [Evaluation script](evaluate.py)
