import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from agents.rollout_buffer import RolloutBuffer
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from models.offloading_policy import OffloadingPolicyNetwork


class MAPPOAgent:
    def __init__(self, network: OffloadingPolicyNetwork, device: str = "cpu"):
        self.network = network.to(device)
        self.device = device

        named_params = [(n, p) for n, p in self.network.named_parameters() if p.requires_grad]
        actor_prefixes = (
            "actor_critic.cross_attention.",
            "actor_critic.subtask_head.",
            "actor_critic.actor_head.",
            "actor_critic.layer_norm.",
            "power_cond_mlp.",
        )
        critic_prefixes = (
            "actor_critic.critic_head.",
            "actor_critic.cost_power_head.",
            "actor_critic.cost_trust_head.",
        )

        self.shared_trunk_params = []
        self.actor_head_params = []
        self.critic_head_params = []
        for name, param in named_params:
            if any(name.startswith(prefix) for prefix in critic_prefixes):
                self.critic_head_params.append(param)
            elif any(name.startswith(prefix) for prefix in actor_prefixes):
                self.actor_head_params.append(param)
            else:
                self.shared_trunk_params.append(param)

        if not self.shared_trunk_params or not self.actor_head_params or not self.critic_head_params:
            raise RuntimeError("failed to build parameter groups for MAPPOAgent")

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.shared_trunk_params, "lr": getattr(TC, "LR_ACTOR", 3e-4)},
                {"params": self.actor_head_params, "lr": getattr(TC, "LR_ACTOR", 3e-4)},
                {"params": self.critic_head_params, "lr": getattr(TC, "LR_CRITIC", 1e-3)},
            ]
        )
        self.scheduler = None
        if TC.USE_LR_DECAY:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=TC.LR_DECAY_RATE,
            )

        self.lambda_power = float(getattr(TC, "DUAL_POWER_INIT", 0.0))
        self.lambda_trust = float(getattr(TC, "DUAL_TRUST_INIT", 0.0))
        if self._benign_trust_mode():
            self.lambda_trust = 0.0
        self.last_update_stats = {}

    @staticmethod
    def _benign_trust_mode() -> bool:
        malicious_ratio = float(max(getattr(Cfg, "MALICIOUS_RATIO", 0.0), 0.0))
        reliable_prob = float(np.clip(getattr(Cfg, "TRUST_RELIABLE_PROB", 1.0), 0.0, 1.0))
        return bool(malicious_ratio <= 1e-12 and reliable_prob >= 1.0 - 1e-12)

    @staticmethod
    def _has_invalid_grad(params) -> bool:
        for param in params:
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                return True
        return False

    def select_action(self, obs_list: List[Dict], deterministic: bool = False) -> Dict:
        with torch.no_grad():
            (
                subtask_actions,
                target_actions,
                power_actions,
                log_probs,
                values,
                cost_power_values,
                cost_trust_values,
            ) = self.network.get_action_and_value(obs_list, deterministic=deterministic, device=self.device)

        subtask_actions_np = torch.atleast_1d(subtask_actions).cpu().numpy().astype(int).flatten()
        target_actions_np = torch.atleast_1d(target_actions).cpu().numpy().astype(int).flatten()
        power_actions_np = torch.atleast_1d(power_actions).cpu().numpy().flatten()

        actions = []
        for i in range(len(obs_list)):
            obs_stamp = obs_list[i].get("obs_stamp")
            actions.append(
                {
                    "subtask": int(subtask_actions_np[i]),
                    "target": int(target_actions_np[i]),
                    "power": float(power_actions_np[i]),
                    **({"obs_stamp": int(obs_stamp)} if obs_stamp is not None else {}),
                }
            )

        return {
            "actions": actions,
            "log_probs": log_probs.cpu().numpy(),
            "values": values.cpu().squeeze(-1).numpy(),
            "cost_power_values": cost_power_values.cpu().squeeze(-1).numpy(),
            "cost_trust_values": cost_trust_values.cpu().squeeze(-1).numpy(),
        }

    def _forward_all_values(self, obs_list: List[Dict]):
        with torch.no_grad():
            inputs = self.network.prepare_inputs(obs_list, self.device)
            _, _, _, _, values, cost_power_values, cost_trust_values = self.network.forward(
                node_x=inputs["node_x"],
                adj=inputs["adj"],
                status=inputs["status"],
                location=inputs["location"],
                L_fwd=inputs["L_fwd"],
                L_bwd=inputs["L_bwd"],
                data_matrix=inputs["data_matrix"],
                delta=inputs["delta"],
                resource_ids=inputs["resource_ids"],
                resource_raw=inputs["resource_raw"],
                subtask_index=inputs["subtask_index"],
                action_mask=inputs["action_mask"],
                subtask_mask=inputs.get("subtask_mask"),
                node_valid_mask=inputs.get("node_valid_mask"),
                task_mask=inputs["task_mask"],
                rate_prev=inputs.get("rate_prev"),
                serving_rsu_onehot=inputs.get("serving_rsu_onehot"),
                global_state=inputs.get("global_state"),
            )
        return (
            values.cpu().squeeze(-1).numpy(),
            cost_power_values.cpu().squeeze(-1).numpy(),
            cost_trust_values.cpu().squeeze(-1).numpy(),
        )

    def get_value(self, obs_list: List[Dict]) -> np.ndarray:
        values, _, _ = self._forward_all_values(obs_list)
        return values

    def get_cost_values(self, obs_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        _, cost_power_values, cost_trust_values = self._forward_all_values(obs_list)
        return cost_power_values, cost_trust_values

    def evaluate_actions(
        self, obs_list: List[Dict], actions: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        subtask_values = []
        for obs, act in zip(obs_list, actions):
            sval = int(act.get("subtask", obs.get("subtask_index", 0))) if isinstance(act, dict) else int(obs.get("subtask_index", 0))
            subtask_values.append(max(sval, 0))

        subtask_actions = torch.tensor(subtask_values, dtype=torch.long, device=self.device)
        target_actions = torch.tensor([a["target"] for a in actions], dtype=torch.long, device=self.device)
        power_actions = torch.tensor([a["power"] for a in actions], dtype=torch.float32, device=self.device)
        log_probs, entropy, values, cost_power_values, cost_trust_values, aux = self.network.evaluate_actions(
            obs_list=obs_list,
            subtask_actions=subtask_actions,
            target_actions=target_actions,
            power_actions=power_actions,
            device=self.device,
            return_aux=True,
        )
        return (
            log_probs,
            values.squeeze(-1),
            cost_power_values.squeeze(-1),
            cost_trust_values.squeeze(-1),
            entropy,
            aux,
        )

    def update_duals(self, mean_cost_power: float, mean_cost_trust: float):
        lr = float(getattr(TC, "DUAL_LR", 0.02))
        lam_max = float(getattr(TC, "DUAL_MAX", 5.0))
        budget_power = float(getattr(TC, "COST_BUDGET_POWER", 0.20))
        budget_trust = float(getattr(TC, "COST_BUDGET_TRUST", 0.35))
        self.lambda_power = float(np.clip(self.lambda_power + lr * (float(mean_cost_power) - budget_power), 0.0, lam_max))
        if self._benign_trust_mode():
            self.lambda_trust = 0.0
        else:
            self.lambda_trust = float(np.clip(self.lambda_trust + lr * (float(mean_cost_trust) - budget_trust), 0.0, lam_max))

    def update(self, buffer: RolloutBuffer, batch_size: int = 64) -> float:
        total_loss = 0.0
        total_entropy = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_value_norm = 0.0
        total_cost_power_value = 0.0
        total_cost_trust_value = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_grad_norm = 0.0
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

        early_stop = False
        target_kl = getattr(TC, "TARGET_KL", None)
        kl_stop_mult = float(max(getattr(TC, "TARGET_KL_STOP_MULT", 1.5), 1.0))
        critic_soft_mask_enabled = bool(getattr(TC, "CRITIC_SOFT_ACTIVE_MASK", True))
        critic_inactive_weight = float(np.clip(getattr(TC, "CRITIC_INACTIVE_SAMPLE_WEIGHT", 0.2), 0.0, 1.0))

        for epoch_idx in range(TC.PPO_EPOCH):
            if early_stop:
                break
            ppo_epochs_executed += 1
            for batch_idx, batch in enumerate(buffer.get_batches(batch_size, active_only=False)):
                obs_list = batch["obs_list"]
                actions = batch["actions"]
                old_log_probs = torch.tensor(batch["old_log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
                cost_power_advantages = torch.tensor(batch["cost_power_advantages"], dtype=torch.float32, device=self.device)
                cost_power_returns = torch.tensor(batch["cost_power_returns"], dtype=torch.float32, device=self.device)
                old_cost_power_values = torch.tensor(batch["old_cost_power_values"], dtype=torch.float32, device=self.device)
                cost_trust_advantages = torch.tensor(batch["cost_trust_advantages"], dtype=torch.float32, device=self.device)
                cost_trust_returns = torch.tensor(batch["cost_trust_returns"], dtype=torch.float32, device=self.device)
                old_cost_trust_values = torch.tensor(batch["old_cost_trust_values"], dtype=torch.float32, device=self.device)
                active_masks = torch.tensor(batch["active_masks"], dtype=torch.float32, device=self.device)
                old_values = torch.tensor(batch["old_values"], dtype=torch.float32, device=self.device)
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
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), TC.MAX_GRAD_NORM)
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
                total_grad_norm += float(grad_norm) if grad_norm is not None else 0.0
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
                    early_stop_batch_idx = int(batch_idx)
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
            "grad_norm": total_grad_norm / num_updates,
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

    def decay_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def save(self, path: str):
        payload = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lambda_power": float(self.lambda_power),
            "lambda_trust": float(self.lambda_trust),
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, path)

    def load(self, path: str, restore_optimizer: bool = True, restore_scheduler: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"], strict=False)
        self.lambda_power = float(checkpoint.get("lambda_power", self.lambda_power))
        self.lambda_trust = float(checkpoint.get("lambda_trust", self.lambda_trust))
        if restore_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        if restore_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
