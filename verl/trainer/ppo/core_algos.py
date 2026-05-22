# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    use_dr_grpo=False,
    use_grpopp=False,
    grpopp_config={},
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    assert sum([use_dr_grpo, use_grpopp]) <= 1
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if use_dr_grpo:
                scores[i] = scores[i] - id2mean[index[i]]
            elif norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        if use_grpopp:
            valid_length_list = response_mask.sum(dim=1).detach().cpu().numpy().tolist()
            mean_length = float(np.mean(valid_length_list))
            length_reciprocal_list = []
            for length in valid_length_list:
                length_reciprocal_list.append(
                    1 / ((length / mean_length) ** grpopp_config["alpha"])
                )
            length_reciprocal_mean = float(np.mean(length_reciprocal_list))
            for i in range(bsz):
                scores[i] = scores[i] * (length_reciprocal_list[i] / length_reciprocal_mean)

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_variance_adaptive_cliprange(current_probs, ratio, response_mask, variance_alpha,
                                        variance_base_clip, variance_clip_min, variance_clip_max,
                                        num_bins=5):
    """
    ACPO: Compute adaptive clip range based on importance sampling ratio variance.

    Args:
        current_probs (torch.Tensor): Current policy probabilities, shape (batch_size, response_length)
        ratio (torch.Tensor): Importance sampling ratio, shape (batch_size, response_length)
        response_mask (torch.Tensor): Response mask, shape (batch_size, response_length)
        variance_alpha (float): Variance modulation factor
        variance_base_clip (float): Base clip value
        variance_clip_min (float): Minimum clip value
        variance_clip_max (float): Maximum clip value
        num_bins (int): Number of equal-width probability bins (default: 5)

    Returns:
        tuple: (adaptive_cliprange_low, adaptive_cliprange_high, ratio_mean_list, ratio_std_list)
    """
    # Initialize clip range tensors
    adaptive_cliprange_low = torch.full_like(ratio, variance_base_clip)
    adaptive_cliprange_high = torch.full_like(ratio, variance_base_clip)

    # Dynamically generate equal-width probability intervals based on num_bins
    bin_width = 1.0 / num_bins
    prob_intervals = [(i * bin_width, (i + 1) * bin_width) for i in range(num_bins)]
    
    ratio_mean_list = []
    ratio_std_list = []
    
    for i, (lower_bound, upper_bound) in enumerate(prob_intervals):
        # Determine mask for current interval; last interval includes upper bound (1.0)
        if i == num_bins - 1:
            interval_mask = (current_probs >= lower_bound) & (current_probs <= upper_bound) & response_mask.bool()
        else:
            interval_mask = (current_probs >= lower_bound) & (current_probs < upper_bound) & response_mask.bool()
        
        # If there are tokens in this interval
        if interval_mask.sum() > 0:
            # Get ratio values for this interval
            interval_ratios = ratio[interval_mask]
            
            # Compute variance and standard deviation
            ratio_mean = interval_ratios.mean()
            ratio_var = torch.var(interval_ratios, unbiased=False)
            ratio_std = torch.sqrt(ratio_var + 1e-8)  # Add small constant for numerical stability

            ratio_mean_list.append(ratio_mean.item())
            ratio_std_list.append(ratio_std.item())
            
            # Compute adaptive clip value: base_clip + alpha * std
            adaptive_clip = variance_base_clip + variance_alpha * ratio_std.item()
            
            # Apply clip value bounds
            adaptive_clip = max(variance_clip_min, min(variance_clip_max, adaptive_clip))
            
            # Apply computed clip value to corresponding positions
            adaptive_cliprange_low[interval_mask] = adaptive_clip
            adaptive_cliprange_high[interval_mask] = adaptive_clip
    
    return adaptive_cliprange_low, adaptive_cliprange_high, ratio_mean_list, ratio_std_list


def compute_entropy_selection_mask(entropy, response_mask, entropy_k_ratio, select_top=True):
    """
    Compute mask to select top-k or bottom-k entropy tokens.

    Args:
        entropy (torch.Tensor): Token-level entropy, shape (batch_size, response_length)
        response_mask (torch.Tensor): Response mask, shape (batch_size, response_length)
        entropy_k_ratio (float): Ratio of tokens to select (e.g., 0.2 for 20%)
        select_top (bool): If True, select top-k (highest entropy); if False, select bottom-k (lowest entropy)

    Returns:
        torch.Tensor: Selection mask, shape (batch_size, response_length)
    """
    batch_size, response_length = entropy.shape
    selection_mask = torch.zeros_like(response_mask, dtype=torch.float32)

    for i in range(batch_size):
        # Get valid token indices for this sample
        valid_mask = response_mask[i].bool()
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            continue

        # Number of tokens to select
        num_select = max(1, int(num_valid * entropy_k_ratio))

        # Get entropy values for valid tokens
        valid_entropy = entropy[i][valid_mask]

        # Get indices of top-k or bottom-k entropy tokens
        if select_top:
            # Select tokens with highest entropy
            _, selected_indices = torch.topk(valid_entropy, num_select, largest=True)
        else:
            # Select tokens with lowest entropy
            _, selected_indices = torch.topk(valid_entropy, num_select, largest=False)

        # Map back to original indices
        valid_indices = torch.where(valid_mask)[0]
        original_indices = valid_indices[selected_indices]

        # Set mask
        selection_mask[i, original_indices] = 1.0

    return selection_mask


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    use_dr_grpo=False,
    use_grpopp=False,
    grpopp_config={},
    use_cispo=False,
    clip_ratio_is_high=0.45,
    clip_ratio_is_low=1.0,
    use_variance_adaptive_clip=False,
    variance_alpha=1.0,
    variance_base_clip=0.2,
    variance_clip_min=0.05,
    variance_clip_max=0.5,
    variance_num_bins=5,
    use_entropy_top_k=False,
    use_entropy_bottom_k=False,
    entropy_k_ratio=0.2,
    entropy=None,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        use_cispo (bool, optional):
            Whether to use the CISPO loss. See https://www.arxiv.org/pdf/2506.13585.
            CISPO only constrains the gradient scale instead of clipping off tokens in PPO.
            Defaults to False.
        clip_ratio_is_high (float, optional):
            Upper bound range for CISPO importance sampling weight.
            Defaults to 0.45.
        clip_ratio_is_low (float, optional):
            Lower bound range for CISPO importance sampling weight.
            Defaults to 1.0.
        use_variance_adaptive_clip (bool, optional):
            Whether to use ACPO (variance-based adaptive clipping).
            When enabled, clip ranges are dynamically adjusted based on the variance
            of importance sampling ratios within each probability interval.
            Defaults to False.
        variance_alpha (float, optional):
            Variance modulation factor for ACPO. The adaptive clip is computed as:
            clip = base_clip + alpha * std(ratio). Defaults to 1.0.
        variance_base_clip (float, optional):
            Base clip value for ACPO. Defaults to 0.2.
        variance_clip_min (float, optional):
            Minimum clip value for ACPO. Defaults to 0.05.
        variance_clip_max (float, optional):
            Maximum clip value for ACPO. Defaults to 0.5.
        use_entropy_top_k (bool, optional):
            Whether to only update tokens with highest entropy (top-k).
            When enabled, only the top entropy_k_ratio tokens are updated.
            Defaults to False.
        use_entropy_bottom_k (bool, optional):
            Whether to only update tokens with lowest entropy (bottom-k).
            When enabled, only the bottom entropy_k_ratio tokens are updated.
            Defaults to False.
        entropy_k_ratio (float, optional):
            Ratio of tokens to select for entropy-based selection (e.g., 0.2 for 20%).
            Defaults to 0.2.
        entropy (torch.Tensor, optional):
            Token-level entropy, shape (batch_size, response_length).
            Required when use_entropy_top_k or use_entropy_bottom_k is True.
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ACPO: Adaptive Clipping based on variance of importance sampling ratios
    if use_variance_adaptive_clip:
        current_probs = torch.exp(log_prob)
        adaptive_cliprange_low, adaptive_cliprange_high, _, _ = compute_variance_adaptive_cliprange(
            current_probs, ratio, response_mask, variance_alpha, variance_base_clip,
            variance_clip_min, variance_clip_max, num_bins=variance_num_bins
        )
    else:
        # Use fixed clip ranges
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        adaptive_cliprange_low = cliprange_low
        adaptive_cliprange_high = cliprange_high

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - adaptive_cliprange_low, 1 + adaptive_cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # CISPO: Clipped Importance Sampling Policy Optimization
    # See https://www.arxiv.org/pdf/2506.13585
    if use_cispo:
        ratio = ratio.detach()
        importance_sampling_weight = torch.clamp(
            ratio,
            max=1 + clip_ratio_is_high,
            min=1 - clip_ratio_is_low
        )
        pos_adv_mask = (advantages > 0) & (
            ratio > 1 + cliprange_high
        )
        neg_adv_mask = (advantages < 0) & (
            ratio < 1 - cliprange_low
        )
        adv_mask = ~(pos_adv_mask | neg_adv_mask)
        pg_losses = -advantages * log_prob * importance_sampling_weight * adv_mask

    # Entropy-based selective token update
    # Only update tokens with highest (top-k) or lowest (bottom-k) entropy
    if use_entropy_top_k or use_entropy_bottom_k:
        assert entropy is not None, "entropy must be provided when use_entropy_top_k or use_entropy_bottom_k is True"
        assert not (use_entropy_top_k and use_entropy_bottom_k), "Cannot use both use_entropy_top_k and use_entropy_bottom_k"

        # Compute selection mask based on entropy
        select_top = use_entropy_top_k  # True for top-k, False for bottom-k
        entropy_selection_mask = compute_entropy_selection_mask(
            entropy=entropy,
            response_mask=response_mask,
            entropy_k_ratio=entropy_k_ratio,
            select_top=select_top
        )

        # Apply entropy selection mask to response_mask
        # This will only compute loss on selected tokens
        response_mask = response_mask * entropy_selection_mask

    if use_dr_grpo or use_grpopp:
        pg_loss = verl_F.masked_mean_allavg(pg_losses, response_mask)
    else:
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_ar_reweight_advantage(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    ar_alpha: float = 0.3,
    ar_tau: float = 0.7,
    neg_adv_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute reweighted advantages for AR (Advantage Reweighting) method.

    AR attenuates gradients from low-probability tokens while emphasizing
    parameter updates driven by high-probability tokens.

    Based on: "Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs"
    https://arxiv.org/abs/2505.12929

    The reweighting formula is: A_new = (ar_alpha * prob + ar_tau) * A

    Args:
        advantages (torch.Tensor): Original advantages, shape (batch_size, response_length)
        log_probs (torch.Tensor): Log probabilities of tokens, shape (batch_size, response_length)
        ar_alpha (float): Alpha parameter for linear reweighting. Defaults to 0.3.
        ar_tau (float): Tau parameter for linear reweighting (typically 1 - ar_alpha). Defaults to 0.7.
        neg_adv_weight (float): Weight multiplier for negative advantages. Defaults to 1.0.

    Returns:
        torch.Tensor: Reweighted advantages, shape (batch_size, response_length)
    """
    # Convert log probs to probs
    probs = torch.exp(log_probs)

    # Apply linear reweighting: (ar_alpha * prob + ar_tau) * advantage
    reweight_advantage = (ar_alpha * probs + ar_tau) * advantages

    # Separate positive and negative advantages
    neg_advantage = torch.where(advantages > 0, torch.zeros_like(advantages), reweight_advantage)
    pos_advantage = torch.where(advantages > 0, reweight_advantage, torch.zeros_like(advantages))

    # Apply negative advantage weight
    neg_advantage = neg_advantage * neg_adv_weight

    # Combine reweighted advantages
    reweight_advantage = pos_advantage + neg_advantage

    return reweight_advantage


def compute_lopti_mask(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    prob_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute masks for Lopti (Low-Probability Token Isolation) method.

    Lopti separates updates for low-probability and high-probability tokens
    to prevent low-probability tokens from dominating gradient updates.

    Based on: "Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs"
    https://arxiv.org/abs/2505.12929

    Args:
        log_probs (torch.Tensor): Log probabilities of tokens, shape (batch_size, response_length)
        response_mask (torch.Tensor): Mask for valid response tokens, shape (batch_size, response_length)
        prob_threshold (float): Probability threshold (eta in paper).
            Positive values: low-prob tokens are those with prob < threshold
            Negative values: low-prob tokens are those with prob < |threshold|
            Defaults to 0.5.

    Returns:
        tuple: (low_prob_mask, high_prob_mask) where each mask indicates which tokens to update
    """
    # Convert log probs to probs
    probs = torch.exp(log_probs)

    abs_threshold = abs(prob_threshold)

    # Create masks for low and high probability tokens
    low_prob_mask = (probs < abs_threshold) & response_mask.bool()
    high_prob_mask = (probs >= abs_threshold) & response_mask.bool()

    # Convert to float for computation
    low_prob_mask = low_prob_mask.float()
    high_prob_mask = high_prob_mask.float()

    return low_prob_mask, high_prob_mask
