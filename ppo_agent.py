"""
PPO agent (Section 10 of implementation_v2).

"Algorithm: PPO. ... its clipped objective provides training stability, so the
window-based (FLX) mechanism of DeepRMSA is not needed."

A shared-trunk actor-critic network with action masking: the infeasible
(route, core) options identified by the environment are masked out of the policy
logits before sampling, so every executed allocation is physically secure.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs):
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @staticmethod
    def _masked_logits(logits, mask):
        # mask: 1.0 for feasible actions, 0.0 otherwise.
        return torch.where(mask > 0, logits, torch.full_like(logits, NEG_INF))

    def distribution(self, obs, mask):
        logits, value = self.forward(obs)
        return Categorical(logits=self._masked_logits(logits, mask)), value


class PPOAgent:
    def __init__(self, obs_dim, action_dim, cfg, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, action_dim, cfg.drl.hidden_sizes).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.drl.lr)
        self.action_dim = action_dim

    # ---- acting --------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, mask, greedy: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value = self.net.distribution(obs_t, mask_t)
        if greedy:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    # ---- GAE -----------------------------------------------------------------
    def _compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        gamma, lam = self.cfg.drl.gamma, self.cfg.drl.gae_lambda
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    # ---- learning ------------------------------------------------------------
    def update(self, batch, last_value: float):
        obs = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.asarray(batch["mask"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(batch["action"]), dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(np.asarray(batch["logp"]), dtype=torch.float32, device=self.device)

        rewards = np.asarray(batch["reward"], dtype=np.float32)
        values = np.asarray(batch["value"], dtype=np.float32)
        dones = np.asarray(batch["done"], dtype=np.float32)
        adv_np, ret_np = self._compute_gae(rewards, values, dones, last_value)
        adv = torch.as_tensor(adv_np, device=self.device)
        returns = torch.as_tensor(ret_np, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        mb = self.cfg.drl.minibatch_size
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": 0}

        for _ in range(self.cfg.drl.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                b = idx[start:start + mb]
                dist, value = self.net.distribution(obs[b], masks[b])
                logp = dist.log_prob(actions[b])
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - self.cfg.drl.clip_eps,
                                    1 + self.cfg.drl.clip_eps) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value - returns[b]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = (policy_loss
                        + self.cfg.drl.value_coef * value_loss
                        - self.cfg.drl.entropy_coef * entropy)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["n"] += 1

        k = max(stats["n"], 1)
        return {"policy_loss": stats["policy_loss"] / k,
                "value_loss": stats["value_loss"] / k,
                "entropy": stats["entropy"] / k}

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
