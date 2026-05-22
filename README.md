# ACPO

This repository contains the implementation for **Adaptive Clip Policy Optimization (ACPO)** from the paper **"What are Key Factors for Updates in RL for LLM Reasoning?"**

ACPO is built on top of [verl](https://github.com/volcengine/verl). It adds variance-adaptive clipping for GRPO/RLVR updates, together with the baselines used in the paper: DAPO, CISPO, AR-Lopti, and entropy-based selective token updates.

## Installation

```bash
git clone https://github.com/Control-derek/ACPO.git
cd ACPO
bash setup.sh
conda activate verl
```

`setup.sh` creates a conda environment, installs this package in editable mode, and installs the training dependencies used by the released scripts.

## Data

The math scripts expect ORZ-style parquet files under `DATA_DIR`:

```text
train.parquet
test1.parquet
test2.parquet
test_amc.parquet
test_minerva.parquet
test_aime2025.parquet
```

By default `DATA_DIR=~/data/open_reasoner_zero_nochat`. You can prepare these files with the included preprocessing script after downloading the public ORZ/evaluation data sources:

```bash
git clone https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero.git ~/Open-Reasoner-Zero
git clone https://github.com/sail-sg/understand-r1-zero.git ~/understand-r1-zero
python examples/data_preprocess/open_reasoner_zero_nochat.py
```

## Training

Main ACPO runs for Qwen2.5-7B on ORZ-57K:

```bash
# Near on-policy: 2 updates per rollout
bash scripts/acpo/math_qwen_7b_acpo_near_onpolicy.sh

# Off-policy: 16 updates per rollout
bash scripts/acpo/math_qwen_7b_acpo_offpolicy.sh
```

The scripts default to console logging only. To enable W&B, pass your own project settings explicitly:

```bash
LOGGER="['console','wandb']" WANDB_PROJECT=acpo WANDB_ENTITY=<your-entity> \
  bash scripts/acpo/math_qwen_7b_acpo_offpolicy.sh
```

Common overrides:

```bash
DATA_DIR=/path/to/open_reasoner_zero_nochat \
OUTPUT_ROOT=/path/to/outputs \
N_GPUS=8 \
bash scripts/acpo/math_qwen_7b_acpo_offpolicy.sh
```

## Main Configs

The paper uses two off-policy regimes with the same base setup:

| Regime | Updates/rollout | Train batch | PPO mini batch |
| --- | ---: | ---: | ---: |
| Near on-policy | 2 | 128 | 64 |
| Off-policy | 16 | 256 | 16 |

Shared settings: `Qwen/Qwen2.5-7B`, learning rate `1e-6`, rollout `n=10`, validation `n=8`, max prompt length `1024`, max response length `3072`, total epochs `15`, no KL loss, no entropy bonus, `seq-mean-token-mean` loss aggregation, DAPO clipping `eps_low=0.2`, `eps_high=0.3`.

ACPO main setting: `use_variance_adaptive_clip=True`, `variance_alpha=3.0`, `variance_base_clip=0.2`, `variance_clip_min=0.0`, `variance_clip_max=3.0`, `variance_num_bins=5`.

Baseline scripts:

```bash
bash scripts/acpo/math_qwen_7b_dapo_offpolicy.sh
bash scripts/acpo/math_qwen_7b_cispo_offpolicy.sh
bash scripts/acpo/math_qwen_7b_ar_lopti_offpolicy.sh
bash scripts/acpo/math_qwen_7b_entropy_top_offpolicy.sh
bash scripts/acpo/math_qwen_7b_entropy_bottom_offpolicy.sh
```

Near-on-policy versions are provided with the same names ending in `_near_onpolicy.sh`.

## Acknowledgement

This codebase is a research fork of [verl](https://github.com/volcengine/verl). We thank the verl authors and contributors for the open-source RLHF/RLVR infrastructure. The original code is licensed under Apache-2.0, and this repository keeps the same license.

## Citation

If you use this code, please cite:

```bibtex
@misc{wang2026keyfactorsrlreasoning,
  title  = {What are Key Factors for Updates in RL for LLM Reasoning?},
  author = {Peidong Wang and Demi Wang and Xufang Luo and Jiahang Xu and Xiaocui Yang and Shi Feng and Yuqing Yang and Dongsheng Li},
  year   = {2026}
}
```
