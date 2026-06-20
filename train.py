"""
Training and evaluation driver (Sections 10-11 of implementation_v2).

  * Offline-then-online: pre-train PPO on the simulator, then fine-tune online.
  * Train on dynamically generated traffic; the policy is updated continuously
    from network feedback.
  * Evaluation protocol: test on held-out seeds, and run a robustness test that
    shifts the traffic distribution mid-run.
  * Metrics: blocking probability (primary), fragmentation, spectrum
    utilisation, and QKD (triplet) success rate.

Run:  python train.py            # full schedule from config
      python train.py --smoke    # tiny run for a quick sanity check
"""

import argparse
from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from config import Config
from environment import QKDSSEONEnv
from ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
def collect_rollout(env: QKDSSEONEnv, agent: PPOAgent, n_steps: int, state):
    obs, done = state
    buf = defaultdict(list)
    for _ in range(n_steps):
        mask = env.action_mask()
        if mask.sum() == 0:
            # No feasible (route, core): the request will block whatever we do.
            # Use an all-ones mask so the policy distribution stays valid and the
            # stored log-prob / value are consistent with the update step.
            mask = np.ones(env.action_dim, dtype=bool)
        action, logp, value = agent.act(obs, mask)
        next_obs, reward, done, _ = env.step(action)

        buf["obs"].append(obs)
        buf["mask"].append(mask.astype(np.float32))
        buf["action"].append(action)
        buf["logp"].append(logp)
        buf["value"].append(value)
        buf["reward"].append(reward)
        buf["done"].append(float(done))

        obs = next_obs
        if done:
            obs = env.reset(seed=int(np.random.randint(1_000_000)))
            done = False

    # bootstrap value of the final state
    mask = env.action_mask()
    if mask.sum() == 0:
        last_value = 0.0
    else:
        _, _, last_value = agent.act(obs, mask)
    return buf, last_value, (obs, done)


def train_phase(env: QKDSSEONEnv, agent: PPOAgent, total_requests: int,
                label: str):
    state = (env.reset(seed=env.cfg.seed), False)
    collected = 0
    while collected < total_requests:
        n = min(env.cfg.drl.rollout_requests, total_requests - collected)
        buf, last_value, state = collect_rollout(env, agent, n, state)
        stats = agent.update(buf, last_value)
        collected += n
        served = sum(1 for r in buf["reward"] if r > 0)
        print(f"[{label}] {collected:>8d}/{total_requests} req | "
              f"served/req={served / n:.3f} | "
              f"pi_loss={stats['policy_loss']:+.3f} "
              f"v_loss={stats['value_loss']:.3f} ent={stats['entropy']:.3f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(cfg: Config, agent: PPOAgent, seeds, n_requests: int,
             nonuniform_shift_at: int = None) -> Dict[str, float]:
    """Greedy roll-out on held-out seeds. If `nonuniform_shift_at` is set, the
    traffic distribution is switched to non-uniform after that many requests
    (the robustness test)."""
    agg = defaultdict(list)
    for seed in seeds:
        env = QKDSSEONEnv(cfg)
        env.cfg.drl.episode_length = n_requests
        obs = env.reset(seed=seed)
        utils, frags = [], []
        for i in range(n_requests):
            if nonuniform_shift_at is not None and i == nonuniform_shift_at:
                env.traffic.set_nonuniform()
            mask = env.action_mask()
            if mask.sum() == 0:
                action = 0
            else:
                action, _, _ = agent.act(obs, mask, greedy=True)
            obs, _, done, info = env.step(action)
            utils.append(info["utilization"])
            frags.append(info["fragmentation"])
            if done:
                break
        agg["blocking_probability"].append(env.blocking_probability)
        agg["triplet_success_rate"].append(env.triplet_success_rate)
        agg["utilization"].append(float(np.mean(utils)))
        agg["fragmentation"].append(float(np.mean(frags)))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def evaluate_heuristic(cfg: Config, seeds, n_requests: int) -> Dict[str, float]:
    """KSP first-fit baseline: shortest route, first quantum core, first-fit on
    every channel. Provides a reference for the learned policy."""
    agg = defaultdict(list)
    for seed in seeds:
        env = QKDSSEONEnv(cfg)
        env.cfg.drl.episode_length = n_requests
        env.reset(seed=seed)
        utils, frags = [], []
        for _ in range(n_requests):
            # route 0, quantum core index 0, first-fit on QC/CC/DC
            action = int(np.ravel_multi_index(
                (0, 0, 0, 0, 0), env._action_factors))
            _, _, done, info = env.step(action)
            utils.append(info["utilization"])
            frags.append(info["fragmentation"])
            if done:
                break
        agg["blocking_probability"].append(env.blocking_probability)
        agg["triplet_success_rate"].append(env.triplet_success_rate)
        agg["utilization"].append(float(np.mean(utils)))
        agg["fragmentation"].append(float(np.mean(frags)))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def _print_metrics(title, m):
    print(f"\n{title}")
    print(f"  blocking probability : {m['blocking_probability']:.4f}")
    print(f"  triplet success rate : {m['triplet_success_rate']:.4f}")
    print(f"  spectrum utilisation : {m['utilization']:.4f}")
    print(f"  fragmentation        : {m['fragmentation']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run for a sanity check")
    parser.add_argument("--topology", default=None,
                        choices=["tokyo12", "sixnode"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    if args.topology:
        cfg.net.topology = args.topology
        cfg.net.__post_init__()
    if args.smoke:
        cfg.drl.offline_requests = 2_000
        cfg.drl.online_requests = 1_000
        cfg.drl.rollout_requests = 512
        cfg.drl.episode_length = 100

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = QKDSSEONEnv(cfg)
    agent = PPOAgent(env.obs_dim, env.action_dim, cfg)
    print(f"topology={cfg.net.topology} nodes={env.topo.num_nodes} "
          f"obs_dim={env.obs_dim} action_dim={env.action_dim} "
          f"offered_load={env.traffic.offered_load:.0f} Erlangs")

    # ---- offline pre-training, then online fine-tuning ----------------------
    train_phase(env, agent, cfg.drl.offline_requests, "offline")
    train_phase(env, agent, cfg.drl.online_requests, "online")

    # ---- evaluation on held-out seeds ---------------------------------------
    held_out = [10_001, 10_002, 10_003]
    eval_n = 500 if args.smoke else 5_000
    _print_metrics("PPO  (held-out seeds)",
                   evaluate(cfg, agent, held_out, eval_n))
    _print_metrics("KSP-FF baseline (held-out seeds)",
                   evaluate_heuristic(cfg, held_out, eval_n))

    # ---- robustness: shift the traffic distribution mid-run -----------------
    _print_metrics("PPO  (robustness: traffic shift mid-run)",
                   evaluate(cfg, agent, held_out, eval_n,
                            nonuniform_shift_at=eval_n // 2))

    agent.save("ppo_qkd_sseon.pt")
    print("\nsaved policy -> ppo_qkd_sseon.pt")


if __name__ == "__main__":
    main()



