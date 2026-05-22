# ACPO Experiment Configs

This note records the released math configs recovered from the original `peidong_scripts` directory and the paper appendix.

## Main ACPO

The main paper setting is `alpha=3.0`, `eps_base=0.2`, `eps_min=0.0`, `eps_max=3.0`, with the default `5` probability bins. In the original private scripts this corresponds to:

| Regime | Original script | Public script |
| --- | --- | --- |
| Near on-policy | `math_qwen_7b_grpo_acpo_bsz2x_alpha3_base0.2.sh` | `scripts/acpo/math_qwen_7b_acpo_near_onpolicy.sh` |
| Off-policy | `math_qwen_7b_grpo_acpo_bsz16x_alpha3_base0.2.sh` | `scripts/acpo/math_qwen_7b_acpo_offpolicy.sh` |

The original `alpha2` scripts are ablations, not the main paper configuration.

## Baselines

| Method | Core settings | Public scripts |
| --- | --- | --- |
| DAPO | `eps_low=0.2`, `eps_high=0.3`, no extra method flag | `math_qwen_7b_dapo_{near_onpolicy,offpolicy}.sh` |
| CISPO | `use_cispo=True`, `clip_ratio_is_high=0.45`, `clip_ratio_is_low=1.0` | `math_qwen_7b_cispo_{near_onpolicy,offpolicy}.sh` |
| AR-Lopti | `ar_alpha=0.3`, `ar_tau=0.7`, `ar_neg_adv_weight=1.0`, `lopti_prob_threshold=0.5`, `eps_high=0.24` | `math_qwen_7b_ar_lopti_{near_onpolicy,offpolicy}.sh` |
| High-Entropy | `use_entropy_top_k=True`, `entropy_k_ratio=0.2` | `math_qwen_7b_entropy_top_{near_onpolicy,offpolicy}.sh` |
| Low-Entropy | `use_entropy_bottom_k=True`, `entropy_k_ratio=0.8` | `math_qwen_7b_entropy_bottom_{near_onpolicy,offpolicy}.sh` |

## Shared Math Setup

| Hyperparameter | Near on-policy | Off-policy |
| --- | ---: | ---: |
| Base model | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B` |
| Updates per rollout | `2` | `16` |
| Train batch size | `128` | `256` |
| PPO mini batch size | `64` | `16` |
| Rollout `n` | `10` | `10` |
| Validation `n` | `8` | `8` |
| Learning rate | `1e-6` | `1e-6` |
| Total epochs | `15` | `15` |
| Max prompt length | `1024` | `1024` |
| Max response length | `3072` | `3072` |
| Entropy coefficient | `0.0` | `0.0` |
| KL coefficient | `0.0` | `0.0` |
| Loss aggregation | `seq-mean-token-mean` | `seq-mean-token-mean` |

All public scripts disable W&B by default and write to `outputs/<experiment_name>` unless `OUTPUT_ROOT` or `OUTPUT_DIR` is provided.
