import json
import os
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    LogitsProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from .paths import ADAPTER_DIR, ARTIFACTS_DIR
from .utils import custom_system_prompt


class PixelOnlyProcessor(LogitsProcessor):
    """
    ─ allowed_ids ──────────────────────────┐
      (픽셀 0-9, 줄바꿈, EOS 등)            └── 모델이 뽑을 수 있는 토큰 집합
    min_steps: 최소 n 토큰까지는 EOS 금지
    """
    def __init__(self, allowed_ids, eos_id, min_steps: int = 4):
        self.allowed  = set(allowed_ids)
        self.eos_id   = eos_id
        self.min_steps = min_steps

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1]         # 현재 생성된 토큰 길이
        ban = {self.eos_id} if step < self.min_steps else set()
        mask = torch.full_like(scores, -float("inf"))
        allow = self.allowed - ban
        mask[:, list(allow)] = 0
        return scores + mask

class ARCSolver:
    """Solve ARC grid tasks with a quantized Qwen3-4B model and a LoRA adapter."""

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        model_id = "Qwen/Qwen3-4B"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_data(self, base_dir, multiply_by_300=1):
        """
        ARC 문제 데이터셋을 구성합니다.
        - 모든 json 파일을 한 번만 읽어 메모리에 캐시
        - 캐시된 task들을 multiply_by_300 만큼 반복 활용
        - 최종 리스트는 셔플하여 반환

        Returns:
            pd.DataFrame with columns: ['task', 'train', 'test_input', 'test_output', 'test']
        """
        import random

        all_filenames = sorted([f for f in os.listdir(base_dir) if f.endswith(".json")])
        assert len(all_filenames) == 300, f"Expected 300 tasks, found {len(all_filenames)}"

        # 1. 모든 파일 미리 읽어서 메모리에 캐싱
        task_cache = {}
        for fname in all_filenames:
            with open(os.path.join(base_dir, fname)) as f:
                task_cache[fname] = json.load(f)

        # 2. 캐시된 내용을 활용해서 반복 생성
        dataset = []
        for _ in range(multiply_by_300):
            for fname, task in task_cache.items():
                if len(task) < 4:
                    continue  # 너무 짧은 데이터는 스킵

                # 4개 인덱스 무작위 추출
                grids_idx = random.sample(range(len(task)), k=4)
                train_grids = [task[i] for i in grids_idx[:3]]
                test_grids = [task[i] for i in grids_idx[3:]]

                test_inputs = [{'input': grid['input']} for grid in test_grids]
                test_outputs = [grid['output'] for grid in test_grids]
                combined_tests = [
                    {'input': inp['input'], 'output': out}
                    for inp, out in zip(test_inputs, test_outputs)
                ]

                dataset.append({
                    'task': fname.split(".")[0],
                    'train': train_grids,
                    'test_input': test_inputs,
                    'test_output': test_outputs,
                    'test': combined_tests,
                })

        # 3. 전체 셔플
        rng = np.random.default_rng()
        rng.shuffle(dataset)

        return pd.DataFrame(dataset)

    def parse_grid(self, ids: List[int]):
        # 1) 전체를 문자열로 디코딩해 첫 줄 제거
        txt   = self.tokenizer.decode(ids, skip_special_tokens=True)
        lines = txt.splitlines()
        # 1줄 이상일 때, 첫 줄이 사이즈라면 제거
        try:
            if "*" in lines[0]:
                _ = [int(x) for x in lines[0].split("*")]  # 검증만
                lines = lines[1:]
        except:
            pass

        # 2) ↓ 기존 로직 그대로
        grid = []
        row = []
        inv_map = {tok: pix for pix, tok in enumerate(self.pixel_ids)}

        for idx in ids:
            if idx == self.sep:
                if row:
                    grid.append(row)
                    row = []
            elif idx in inv_map:
                # 오직 pixel 토큰만 append
                row.append(inv_map[idx])
            else:
                # 나머지는 전부 스킵
                continue

        # 마지막 row 추가
        if row:
            grid.append(row)
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_grid_with_size(self, grid: List[List[int]]) -> List[int]:
        """
        "H*W\n" + grid   를 토큰 시퀀스로 반환
        """
        h, w   = len(grid), len(grid[0])
        header = f"{h}*{w}\n"
        ids    = self.tokenizer.encode(header, add_special_tokens=False)
        ids   += self.format_grid(grid)          # 기존 숫자+줄바꿈 토큰
        return ids

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): contains training data, test input

        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + custom_system_prompt, add_special_tokens=False)

        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + "Exampels:" + "\n", add_special_tokens=False)

        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid_with_size(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + "Now solve:" + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)

        user += out_desc

        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis

        return {
            "input_ids": messages,
            "input": input_test_data,
            "train": training_data
        }


    def train(self, train_dataset):
        """
        Train a model with train_dataset.
        Read a project documentation for a description of `examples` and `question`.
        """

        # 1. LoRA 설정
        lora_config = LoraConfig(
            r=8, # 4 or 8
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 2. PEFT 적용
        model = prepare_model_for_kbit_training(self.model)
        model = get_peft_model(model, lora_config)
        self.model = model

        # 3. 학습용 데이터셋 포맷 (여기선 간단히 구성)
        def tokenize_example(example):
            # 1) 프롬프트 생성 (train examples + test input, 템플릿까지)
            prompt_ids = self.format_prompt(example)["input_ids"]
            # 2) 거기에 example['test']의 출력 그리드를 연속으로 붙여준다
            #    (즉 모델이 이 토큰들을 생성하도록 학습)

            test_output_ids = self.format_grid_with_size(example["test"][0]["output"])

            input_ids = prompt_ids + test_output_ids + [self.tokenizer.eos_token_id]
            labels    = [-100] * len(prompt_ids) + test_output_ids + [self.tokenizer.eos_token_id]
            #   => prompt 부분은 loss 계산에서 무시(ignore_index=-100),
            #      뒤의 out_ids 만 학습 신호를 줌
            return {"input_ids": input_ids, "labels": labels}

        tokenized_dataset = [tokenize_example(d) for d in train_dataset]

        # 4. TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(ADAPTER_DIR),
            per_device_train_batch_size=1,
            num_train_epochs=2, # 3 -> 8 -> 11 -> 14
            logging_steps=10,
            save_steps=1000, # TODO: accum 4로 할 때는 250으로 수정
            save_total_limit=12,
            fp16=True,
            remove_unused_columns=False,
            report_to="none",

            gradient_accumulation_steps=8,
            optim="paged_adamw_8bit", # lr(default): 5e-5
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        checkpoint = get_last_checkpoint(str(ADAPTER_DIR)) if ADAPTER_DIR.is_dir() else None
        trainer.train(resume_from_checkpoint=checkpoint)
        model.save_pretrained(str(ADAPTER_DIR))

    def predict(self, examples, questions_input, num_samples=1, temperature=0.8, top_p=0.9):
        # ── 0. prompt ────────────────────────────────────────────────
        dp        = {"train": examples, "test": [{"input": questions_input}]}
        p         = self.format_prompt(dp)
        input_ids = torch.tensor(p["input_ids"], device=self.device)[None, :]
        N_prompt  = input_ids.size(1)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # ── 1. 공통 generation config ────────────────────────────────
        gen_cfg = GenerationConfig(
            do_sample      = True,
            temperature    = temperature,
            top_p          = top_p,
            max_new_tokens = 150,
            pad_token_id   = self.tokenizer.eos_token_id,
            eos_token_id   = self.tokenizer.eos_token_id,
        )

        processor = PixelOnlyProcessor(
            allowed_ids=self.pixel_ids + [self.sep] + [self.tokenizer.eos_token_id]
                        + self.tokenizer.encode("*", add_special_tokens=False),
            eos_id=self.tokenizer.eos_token_id,
            min_steps=5 # {H}{*}{W}{\n}{#}
        )

        all_preds = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = self.model.generate(
                    input_ids         = input_ids,
                    attention_mask    = attn_mask,
                    generation_config = gen_cfg,
                    logits_processor  = [processor]
                )[0].cpu().tolist()

                gen2_ids   = out[N_prompt:]
                txt2       = self.tokenizer.decode(gen2_ids, skip_special_tokens=True)
                lines2     = txt2.splitlines()

                if not lines2 or "*" not in lines2[0]:
                    print("missing grid header:", lines2[:1])
                    continue

                try:
                    h, w = map(int, lines2[0].split("*"))
                except ValueError:
                    print("header parse fail:", lines2[0])
                    continue

                body_ids = self.tokenizer.encode("\n".join(lines2[1:]),
                                                 add_special_tokens=False)
                grid = self.parse_grid(body_ids)
                if grid and any(len(row) != len(grid[0]) for row in grid):
                    print("inconsistent row widths", lines2[:3])
                    continue
                grid_np = np.array(grid, dtype=int)
                if grid_np.size == 0:
                    print("empty grid", lines2[:3])
                    continue
                if grid_np.shape != (h, w):
                    print("shape mismatch", grid_np.shape, (h, w))
                    continue

                # pad / crop
                fixed = np.zeros((h, w), dtype=int)
                hh, ww = min(grid_np.shape[0], h), min(grid_np.shape[1], w)
                fixed[:hh, :ww] = grid_np[:hh, :ww]

                all_preds.append(fixed)

        # ── 5. fallback ──────────────────────────────────────────────
        if not all_preds:
            h0, w0 = np.array(questions_input).shape
            return np.zeros((h0, w0), dtype=int).tolist()

        # ── 6. majority voting ───────────────────────────────────────
        Hf, Wf = all_preds[0].shape
        final  = np.zeros((Hf, Wf), dtype=int)
        for i in range(Hf):
            for j in range(Wf):
                votes = [g[i, j] for g in all_preds]
                final[i, j] = Counter(votes).most_common(1)[0][0]

        return final.tolist()

    def prepare_evaluation(self):
        """
        Load trained LoRA adapter weights using `load_adapter` and set the model to evaluation mode.
        Assumes base model is already loaded in __init__.

        기존 base model에 load adapter, base model을 full tuning하는게 아니라
        LoRA 방식으로 adapter만 학습시키는 경우 제출을 adapter만 하고 evaluate.py에서 이 함수를 호출했을 때
        init에서 메모리에 로드해둔 base model에 adapter를 붙여서 완전한 모델로 복원한 뒤, predict() 호출하는 방T식
        """
        if not (ADAPTER_DIR / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Train the adapter with python -m arc.train first: {ADAPTER_DIR}")
        self.model.load_adapter(str(ADAPTER_DIR), adapter_name="default")
        self.model.eval()

    def initialize_fresh_adapter(self, adapter_save_path=ARTIFACTS_DIR / "checkpoint-fresh"):
        """
        현재 ARCSolver 객체의 새로운 self.model(base model)을 기준으로 초기화된 LoRA adapter를 생성 및 저장합니다.
        기존 학습이 전혀 없는 상태입니다.

        Args:
            adapter_save_path (str): 초기화된 adapter 저장 경로

        """
        lora_config = LoraConfig(
            r=4,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model_for_peft = prepare_model_for_kbit_training(self.model)
        peft_model = get_peft_model(model_for_peft, lora_config)

        peft_model.save_pretrained(adapter_save_path)

    def analyze_seq_lengths(self, dataset):
        """
        dataset: List[dict]
            각 dict는 {"train": [...], "test": [...]}

        Returns
        -------
        info: dict
            {
              "n_samples": int,
              "mean": float,
              "max": int,
              "p50": float,
              "p90": float,
              "p99": float,
              "histogram": Counter  # token 길이 → 개수
            }
        """
        lengths = []
        hist = Counter()

        for example in dataset:
            prompt_ids      = self.format_prompt(example)["input_ids"]  # train + test in
            test_output_ids = self.format_grid(example["test"][0]["output"])
            total_len       = len(prompt_ids) + len(test_output_ids)

            lengths.append(total_len)
            hist[total_len] += 1

        arr = np.array(lengths)
        info = {
            "n_samples": len(lengths),
            "mean": float(arr.mean().round(2)),
            "max":  int(arr.max()),
            "p50":  float(np.percentile(arr, 50).round(2)),
            "p90":  float(np.percentile(arr, 90).round(2)),
            "p99":  float(np.percentile(arr, 99).round(2)),
            "histogram": hist,
        }
        return info


if __name__ == "__main__":
    solver = ARCSolver()
