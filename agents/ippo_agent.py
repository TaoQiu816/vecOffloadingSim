from __future__ import annotations

import numpy as np
import torch

from configs.train_config import TrainConfig as TC
from agents.mappo_agent import MAPPOAgent


class IPPOAgent(MAPPOAgent):
    """
    Minimal IPPO baseline on top of the current shared-policy codebase.

    - same observation / feature encoders / hybrid action space
    - decentralized critic path (handled in OffloadingPolicyNetwork via TC.ALGO_MODE)
    - PPO updates grouped by stable agent_id, i.e. each agent trajectory is updated independently
    """

    def update(self, buffer, batch_size: int = 64) -> float:
        total_loss = 0.0
        total_entropy = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_value_norm = 0.0
        total_cost_power_value = 0.0
        total_cost_trust_value = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_grad_norm_preclip = 0.0
        total_grad_norm_postclip = 0.0
        total_value_clip = 0.0
        total_value_target_mean = 0.0
        total_value_target_std = 0.0
        total_value_pred_mean = 0.0
        total_value_pred_std = 0.0
        total_critic_loss_active = 0.0
        total_critic_loss_inactive = 0.0
        total_mask_active = 0.0
        total_mask_count = 0.0
        num_updates = 0
        ppo_epochs_executed = 0
        num_minibatches_executed = 0
        mb_kl_values = []
        mb_kl_max = 0.0
        early_stop_epoch_idx = -1
        early_stop_batch_idx = -1
        active_samples, total_samples = buffer.get_active_stats()
        adv_mean, adv_std = buffer.get_adv_stats()

        if active_samples < max(1, TC.MIN_ACTIVE_SAMPLES):
            self.last_update_stats = {
                "loss": 0.0,
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "entropy_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "value_loss_raw_mean": 0.0,
                "normalized_value_loss": 0.0,
                "cost_power_value_loss": 0.0,
                "cost_trust_value_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
                "grad_norm_preclip": 0.0,
                "grad_norm_postclip": 0.0,
                "active_samples": int(active_samples),
                "total_samples": int(total_samples),
                "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
                "actor_update_active_frac": 0.0,
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "value_pred_mean": 0.0,
                "value_pred_std": 0.0,
                "value_clip_fraction": 0.0,
                "critic_loss_active": 0.0,
                "critic_loss_inactive": 0.0,
                "ppo_epochs_executed": 0,
                "num_minibatches_executed": 0,
                "mb_kl_max": 0.0,
                "mb_kl_p95": 0.0,
                "early_stop_epoch_idx": -1,
                "early_stop_batch_idx": -1,
                "skipped_update_count": 1,
                "early_stop": False,
                "lambda_power": float(self.lambda_power),
                "lambda_trust": float(self.lambda_trust),
            }
            return 0.0

        flat = buffer.flatten_samples(normalize_advantages=False)
        flat_agent_ids = flat["agent_ids"]
        unique_agent_ids = [int(a) for a in np.unique(flat_agent_ids)] if flat_agent_ids.size > 0 else []
        if not unique_agent_ids:
            self.last_update_stats = {"loss": 0.0, "active_samples": int(active_samples), "total_samples": int(total_samples)}
            return 0.0

        early_stop = False
        target_kl = getattr(TC, "TARGET_KL", None)
        kl_stop_mult = float(max(getattr(TC, "TARGET_KL_STOP_MULT", 1.5), 1.0))
        critic_soft_mask_enabled = bool(getattr(TC, "CRITIC_SOFT_ACTIVE_MASK", True))
        critic_inactive_weight = float(np.clip(getattr(TC, "CRITIC_INACTIVE_SAMPLE_WEIGHT", 0.2), 0.0, 1.0))

        def _normalized_subset(values_arr: np.ndarray, masks_arr: np.ndarray) -> np.ndarray:
            out = values_arr.astype(np.float32).copy()
            active_idx = masks_arr > 0.0
            if np.any(active_idx):
                mean = out[active_idx].mean()
                std = out[active_idx].std()
            else:
                mean = out.mean() if out.size > 0 else 0.0
                std = out.std() if out.size > 0 else 0.0
            if std > 1e-8:
                out = (out - mean) / (std + 1e-8)
            return out

        for epoch_idx in range(TC.PPO_EPOCH):
            if early_stop:
                break
            ppo_epochs_executed += 1
            shuffled_agents = np.random.permutation(unique_agent_ids)
            for agent_loop_idx, agent_id in enumerate(shuffled_agents):
                candidate_idx = np.where(flat_agent_ids == int(agent_id))[0]
                if candidate_idx.size == 0:
                    continue
                indices = np.random.permutation(candidate_idx)

                agent_adv = _normalized_subset(flat["advantages"][candidate_idx], flat["active_masks"][candidate_idx])
                agent_cp_adv = _normalized_subset(flat["cost_power_advantages"][candidate_idx], flat["active_masks"][candidate_idx])
                agent_ct_adv = _normalized_subset(flat["cost_trust_advantages"][candidate_idx], flat["active_masks"][candidate_idx])
                adv_map = {int(idx): float(val) for idx, val in zip(candidate_idx.tolist(), agent_adv.tolist())}
                cp_map = {int(idx): float(val) for idx, val in zip(candidate_idx.tolist(), agent_cp_adv.tolist())}
                ct_map = {int(idx): float(val) for idx, val in zip(candidate_idx.tolist(), agent_ct_adv.tolist())}

                num_batches = max(1, len(indices) // batch_size)
                for batch_local_idx in range(num_batches):
                    start_idx = batch_local_idx * batch_size
                    end_idx = min((batch_local_idx + 1) * batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    if len(batch_indices) == 0:
                        continue

                    obs_list = [flat["obs_list"][idx] for idx in batch_indices]
                    actions = [flat["actions"][idx] for idx in batch_indices]
                    old_log_probs = torch.tensor(flat["old_log_probs"][batch_indices], dtype=torch.float32, device=self.device)
                    advantages = torch.tensor([adv_map[int(idx)] for idx in batch_indices], dtype=torch.float32, device=self.device)
                    returns = torch.tensor(flat["returns"][batch_indices], dtype=torch.float32, device=self.device)
                    cost_power_advantages = torch.tensor([cp_map[int(idx)] for idx in batch_indices], dtype=torch.float32, device=self.device)
                    cost_power_returns = torch.tensor(flat["cost_power_returns"][batch_indices], dtype=torch.float32, device=self.device)
                    old_cost_power_values = torch.tensor(flat["old_cost_power_values"][batch_indices], dtype=torch.float32, device=self.device)
                    cost_trust_advantages = torch.tensor([ct_map[int(idx)] for idx in batch_indices], dtype=torch.float32, device=self.device)
                    cost_trust_returns = torch.tensor(flat["cost_trust_returns"][batch_indices], dtype=torch.float32, device=self.device)
                    old_cost_trust_values = torch.tensor(flat["old_cost_trust_values"][batch_indices], dtype=torch.float32, device=self.device)
                    active_masks = torch.tensor(flat["active_masks"][batch_indices], dtype=torch.float32, device=self.device)
                    old_values = torch.tensor(flat["old_values"][batch_indices], dtype=torch.float32, device=self.device)
                    actor_masks = active_masks
                    if critic_soft_mask_enabled:
                        inactive_valid_masks = (active_masks <= 0.0).float()
                        critic_masks = actor_masks + critic_inactive_weight * inactive_valid_masks
                    else:
                        critic_masks = torch.ones_like(active_masks)

                    log_probs, values, cost_power_values, cost_trust_values, entropy, aux = self.evaluate_actions(obs_list, actions)
                    ratio = torch.exp(log_probs - old_log_probs)
                    actor_mask_sum = actor_masks.sum()
                    critic_mask_sum = critic_masks.sum()
                    total_mask_active += float(actor_mask_sum.item())
                    total_mask_count += float(actor_masks.numel())
                    if critic_mask_sum.item() < 1e-6:
                        continue

                    if actor_mask_sum.item() < 1.0:
                        policy_loss = torch.tensor(0.0, device=self.device)
                        entropy_mean = torch.tensor(0.0, device=self.device)
                        entropy_loss = torch.tensor(0.0, device=self.device)
                        approx_kl = torch.tensor(0.0, device=self.device)
                        clip_frac = torch.tensor(0.0, device=self.device)
                    else:
                        lambda_trust = 0.0 if self._benign_trust_mode() else self.lambda_trust
                        lagrangian_adv = advantages - (self.lambda_power * cost_power_advantages) - (lambda_trust * cost_trust_advantages)
                        surr1 = ratio * lagrangian_adv
                        surr2 = torch.clamp(ratio, 1.0 - TC.CLIP_PARAM, 1.0 + TC.CLIP_PARAM) * lagrangian_adv
                        policy_loss = -(torch.min(surr1, surr2) * actor_masks).sum() / actor_mask_sum
                        entropy_mean = (entropy * actor_masks).sum() / actor_mask_sum
                        entropy_loss = -entropy_mean
                        approx_kl = ((old_log_probs - log_probs) * actor_masks).sum() / actor_mask_sum
                        clip_frac = ((torch.abs(ratio - 1.0) > TC.CLIP_PARAM).float() * actor_masks).sum() / actor_mask_sum

                    critic_idx = critic_masks > 0.0
                    masked_returns = returns[critic_idx]
                    masked_values = values[critic_idx]
                    masked_old_values = old_values[critic_idx]
                    value_target_mean = masked_returns.mean() if masked_returns.numel() > 0 else torch.tensor(0.0, device=self.device)
                    value_target_std = masked_returns.std(unbiased=False) if masked_returns.numel() > 0 else torch.tensor(0.0, device=self.device)
                    value_pred_mean = masked_values.mean() if masked_values.numel() > 0 else torch.tensor(0.0, device=self.device)
                    value_pred_std = masked_values.std(unbiased=False) if masked_values.numel() > 0 else torch.tensor(0.0, device=self.device)
                    if TC.USE_VALUE_TARGET_NORM and masked_returns.numel() > 0 and value_target_std.item() > 1e-6:
                        norm_std = value_target_std + 1e-8
                        returns_used = (returns - value_target_mean) / norm_std
                        values_used = (values - value_target_mean) / norm_std
                        old_values_used = (old_values - value_target_mean) / norm_std
                    else:
                        returns_used = returns
                        values_used = values
                        old_values_used = old_values

                    value_diff = values_used - old_values_used
                    if TC.USE_VALUE_CLIP:
                        value_pred_clipped = old_values_used + torch.clamp(value_diff, -TC.VALUE_CLIP_RANGE, TC.VALUE_CLIP_RANGE)
                    else:
                        value_pred_clipped = values_used
                    value_loss_raw = (values_used - returns_used) ** 2
                    value_loss_clip = (value_pred_clipped - returns_used) ** 2
                    value_loss_max = torch.max(value_loss_raw, value_loss_clip) if TC.USE_VALUE_CLIP else value_loss_raw
                    value_loss_raw_mean = (value_loss_max * critic_masks).sum() / critic_mask_sum
                    masked_returns_used = returns_used[critic_idx]
                    returns_var = masked_returns_used.var(unbiased=False) + 1e-8 if masked_returns_used.numel() > 0 else torch.tensor(1.0, device=self.device)
                    value_loss = value_loss_raw_mean / returns_var

                    def _cost_loss(pred, old_pred, target):
                        pred_diff = pred - old_pred
                        if TC.USE_VALUE_CLIP:
                            pred_clipped = old_pred + torch.clamp(pred_diff, -TC.VALUE_CLIP_RANGE, TC.VALUE_CLIP_RANGE)
                        else:
                            pred_clipped = pred
                        raw = (pred - target) ** 2
                        clip = (pred_clipped - target) ** 2
                        return (torch.max(raw, clip) * critic_masks).sum() / critic_mask_sum

                    cost_power_loss = _cost_loss(cost_power_values, old_cost_power_values, cost_power_returns)
                    cost_trust_loss = _cost_loss(cost_trust_values, old_cost_trust_values, cost_trust_returns)

                    active_idx = actor_masks > 0.0
                    inactive_idx = active_masks <= 0.0
                    critic_loss_active = value_loss_max[active_idx].mean() if torch.any(active_idx) else torch.tensor(0.0, device=self.device)
                    critic_loss_inactive = value_loss_max[inactive_idx].mean() if torch.any(inactive_idx) else torch.tensor(0.0, device=self.device)
                    value_clip_frac = (
                        (torch.abs(values_used - old_values_used) > TC.VALUE_CLIP_RANGE).float() * critic_masks
                    ).sum() / critic_mask_sum if TC.USE_VALUE_CLIP else torch.tensor(0.0, device=self.device)

                    actor_loss_total = policy_loss + TC.ENTROPY_COEF * entropy_loss
                    critic_loss_total = (
                        TC.VF_COEF * value_loss
                        + float(getattr(TC, "COST_VF_COEF", 0.5)) * (cost_power_loss + cost_trust_loss)
                    )
                    loss = actor_loss_total + critic_loss_total
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm_preclip = sum(p.grad.data.norm(2).item() ** 2 for p in self.network.parameters() if p.grad is not None) ** 0.5
                    grad_norm_postclip = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TC.MAX_GRAD_NORM)
                    if self._has_invalid_grad(self.network.parameters()):
                        continue
                    self.optimizer.step()

                    total_loss += float(loss.item())
                    total_entropy += float(entropy_mean.item())
                    total_policy += float(policy_loss.item())
                    total_value += float(value_loss_raw_mean.item())
                    total_value_norm += float(value_loss.item())
                    total_cost_power_value += float(cost_power_loss.item())
                    total_cost_trust_value += float(cost_trust_loss.item())
                    total_kl += float(approx_kl.item())
                    total_clip += float(clip_frac.item())
                    total_grad_norm_preclip += float(grad_norm_preclip)
                    total_grad_norm_postclip += float(grad_norm_postclip)
                    total_value_clip += float(value_clip_frac.item())
                    total_value_target_mean += float(value_target_mean.item())
                    total_value_target_std += float(value_target_std.item())
                    total_value_pred_mean += float(value_pred_mean.item())
                    total_value_pred_std += float(value_pred_std.item())
                    total_critic_loss_active += float(critic_loss_active.item())
                    total_critic_loss_inactive += float(critic_loss_inactive.item())
                    num_updates += 1
                    num_minibatches_executed += 1
                    mb_kl = float(approx_kl.item())
                    mb_kl_values.append(mb_kl)
                    mb_kl_max = max(mb_kl_max, mb_kl)
                    if target_kl is not None and target_kl > 0.0 and mb_kl > (target_kl * kl_stop_mult):
                        early_stop = True
                        early_stop_epoch_idx = int(epoch_idx)
                        early_stop_batch_idx = int(num_minibatches_executed - 1)
                        break
                if early_stop:
                    break
            if early_stop:
                break

        mb_kl_p95 = float(np.percentile(mb_kl_values, 95)) if mb_kl_values else 0.0
        if num_updates <= 0:
            self.last_update_stats = {
                "loss": 0.0,
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "entropy_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "value_loss_raw_mean": 0.0,
                "normalized_value_loss": 0.0,
                "cost_power_value_loss": 0.0,
                "cost_trust_value_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
                "active_samples": int(active_samples),
                "total_samples": int(total_samples),
                "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
                "actor_update_active_frac": (total_mask_active / total_mask_count) if total_mask_count > 0 else 0.0,
                "adv_mean": float(adv_mean),
                "adv_std": float(adv_std),
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "value_pred_mean": 0.0,
                "value_pred_std": 0.0,
                "value_clip_fraction": 0.0,
                "critic_loss_active": 0.0,
                "critic_loss_inactive": 0.0,
                "ppo_epochs_executed": int(ppo_epochs_executed),
                "num_minibatches_executed": int(num_minibatches_executed),
                "mb_kl_max": float(mb_kl_max),
                "mb_kl_p95": float(mb_kl_p95),
                "early_stop_epoch_idx": int(early_stop_epoch_idx),
                "early_stop_batch_idx": int(early_stop_batch_idx),
                "skipped_update_count": 0,
                "early_stop": early_stop,
                "lambda_power": float(self.lambda_power),
                "lambda_trust": float(self.lambda_trust),
            }
            return 0.0

        avg_entropy = max(total_entropy / num_updates, 0.0)
        self.last_update_stats = {
            "loss": total_loss / num_updates,
            "entropy": avg_entropy,
            "policy_entropy": avg_entropy,
            "entropy_loss": -avg_entropy,
            "policy_loss": total_policy / num_updates,
            "value_loss": total_value / num_updates,
            "value_loss_raw_mean": total_value / num_updates,
            "normalized_value_loss": total_value_norm / num_updates,
            "cost_power_value_loss": total_cost_power_value / num_updates,
            "cost_trust_value_loss": total_cost_trust_value / num_updates,
            "approx_kl": total_kl / num_updates,
            "clip_fraction": total_clip / num_updates,
            "grad_norm": total_grad_norm_preclip / num_updates,
            "grad_norm_preclip": total_grad_norm_preclip / num_updates,
            "grad_norm_postclip": total_grad_norm_postclip / num_updates,
            "active_samples": int(active_samples),
            "total_samples": int(total_samples),
            "active_ratio": (float(active_samples) / float(total_samples)) if total_samples > 0 else 0.0,
            "actor_update_active_frac": (total_mask_active / total_mask_count) if total_mask_count > 0 else 0.0,
            "adv_mean": float(adv_mean),
            "adv_std": float(adv_std),
            "value_target_mean": total_value_target_mean / num_updates,
            "value_target_std": total_value_target_std / num_updates,
            "value_pred_mean": total_value_pred_mean / num_updates,
            "value_pred_std": total_value_pred_std / num_updates,
            "value_clip_fraction": total_value_clip / num_updates,
            "critic_loss_active": total_critic_loss_active / num_updates,
            "critic_loss_inactive": total_critic_loss_inactive / num_updates,
            "ppo_epochs_executed": int(ppo_epochs_executed),
            "num_minibatches_executed": int(num_minibatches_executed),
            "mb_kl_max": float(mb_kl_max),
            "mb_kl_p95": float(mb_kl_p95),
            "early_stop_epoch_idx": int(early_stop_epoch_idx),
            "early_stop_batch_idx": int(early_stop_batch_idx),
            "skipped_update_count": 0,
            "early_stop": early_stop,
            "lambda_power": float(self.lambda_power),
            "lambda_trust": float(self.lambda_trust),
        }
        return total_loss / num_updates
