# ARC-AGI LoRA

ARC 격자 퍼즐의 입력·출력 예시 세 쌍을 보고 다음 입력의 출력 격자를 예측하는 실험입니다. Qwen3-4B를 4비트로 불러와 LoRA 어댑터를 학습합니다.

## 데이터

`dataset/`에는 과제별 JSON 파일 300개와 입력·출력 격자 269,312쌍이 들어 있습니다. 학습할 때는 각 과제에서 서로 다른 네 쌍을 뽑아 세 쌍을 예시로, 한 쌍을 예측 대상으로 사용합니다.

## 실행

CUDA GPU 환경에서 저장소 루트를 기준으로 실행합니다. 처음 실행할 때 Hugging Face에서 `Qwen/Qwen3-4B`를 내려받습니다.

```bash
python -m pip install -r requirements.txt
python -m arc.train
python evaluate.py
```

학습은 `artifacts/checkpoint-qwen-4B-150/`에 중간 체크포인트와 최종 어댑터를 저장합니다. 다시 실행하면 가장 최근 체크포인트부터 이어서 학습합니다.

평가 스크립트는 같은 데이터에서 샘플링한 문제 10개를 풀고, 예측 격자의 크기와 모든 셀이 정답과 일치한 비율을 출력합니다. 예시 격자만 보려면 `python examples.py`를 실행하세요.
