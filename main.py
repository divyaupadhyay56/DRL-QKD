import numpy as np
import pandas as pd
from collections import Counter

from topology import MCFNetwork
from traffic import TrafficGenerator
from environment import QKDEnvironment
from ppo_agent import PPOAgent
import mcf_resources as mr

# ---------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------
K_PATHS = 15
J_BLOCKS = 3          # candidate free blocks per channel

HOLDING_MEAN = 6.0

LOADS = [90, 120, 150]

EPISODE_SIZE = 1500
NUM_TRAINING_SLOTS = 250

OFFLINE_EPISODES = 10
ONLINE_EPISODES = 10
EVAL_EPISODES = 5

def run_episode(env, agent, episode_seed, training=True):

    env.traffic.rng = np.random.default_rng(episode_seed)

    env.reset()

    blocked = 0
    served = 0

    block_reasons = Counter()

    frag_samples = []
    util_samples = []
    rewards = []

    requests = list(env.traffic.generate(EPISODE_SIZE))

    last_value = 0.0

    for req in requests:

        env.release_expired(req["arrival"])

        routes = env._candidate_routes(req)

        if not routes:
            blocked += 1
            block_reasons["route"] += 1
            rewards.append(-1.0)
            continue

        state = env.build_state(req, routes)       # it gives information of Network and available free blocks to PPO.

        mask = env.action_mask(req, routes)  #It informs PPO which blocks are valid for this request and which ones cannot be selected at all.

        if mask.sum() == 0:
            blocked += 1
            block_reasons[
                env.mask_block_reason(req, routes)
            ] += 1
            rewards.append(-1.0)
            continue

        action, logp, value = agent.act(
            state,
            mask
        )

        reward, is_blocked, info = env.step(
            req,
            routes,
            action
        )

        if training:
            agent.remember(
                state,
                action,
                logp,
                reward,
                value,
                mask
            )

        last_value = value

        rewards.append(reward)

        if info["served"]:
            served += 1

        if is_blocked:
            blocked += 1
            block_reasons[info["block_reason"]] += 1

        frag_samples.append(
            mr.external_fragmentation(
                env.slot_table
            )
        )

        util_samples.append(
            mr.spectrum_utilisation(
                env.slot_table
            )
        )

    if training:
        agent.train(last_value=last_value)

    n = len(requests)

    return {
        "blocking_probability": blocked / n,

        "fragmentation":
            float(np.mean(frag_samples))
            if frag_samples else 0.0,

        "utilisation":
            float(np.mean(util_samples))
            if util_samples else 0.0,

        "qkd_success_rate":
            served / n,

        "mean_reward":
            float(np.mean(rewards))
            if rewards else 0.0,

        "blocked_route": block_reasons["route"] / n,
        "blocked_qc": block_reasons["qc"] / n,
        "blocked_cc": block_reasons["cc"] / n,
        "blocked_dc": block_reasons["dc"] / n,
        "blocked_combination":
            block_reasons["combination"] / n,
    }


def main():

    history = {
        "load": [],
        "phase": [],
        "episode": [],
        "blocking_probability": [],
        "fragmentation": [],
        "utilisation": [],
        "qkd_success_rate": [],
        "mean_reward": [],
        "blocked_route": [],
        "blocked_qc": [],
        "blocked_cc": [],
        "blocked_dc": [],
        "blocked_combination": [],
    }

    eval_summary = {
        "load": [],
        "avg_eval_fragmentation": [],
    }

    for load in LOADS:

        arrival_rate = load / HOLDING_MEAN

        net = MCFNetwork()

        traffic = TrafficGenerator(
            nodes=net.nodes,
            arrival_rate=arrival_rate,
            holding_mean=HOLDING_MEAN
        )

        env = QKDEnvironment(
            net,
            traffic,
            k_paths=K_PATHS,
            total_requests=EPISODE_SIZE,
            num_training_slots=NUM_TRAINING_SLOTS,
            j_blocks=J_BLOCKS
        )

        agent = PPOAgent(
            env.state_size,
            env.action_size
        )

        print("\n")
        print("=" * 60)
        print(f"Training for Load = {load} Erlangs")
        print("=" * 60)

        print(
            "state_size =",
            env.state_size,
            "action_size =",
            env.action_size
        )

        def log(phase, ep, m):

            history["load"].append(load)
            history["phase"].append(phase)
            history["episode"].append(ep)

            for key in (
                "blocking_probability",
                "fragmentation",
                "utilisation",
                "qkd_success_rate",
                "mean_reward",
                "blocked_route",
                "blocked_qc",
                "blocked_cc",
                "blocked_dc",
                "blocked_combination"
            ):
                history[key].append(m[key])

            print(
                f"[{phase}] ep {ep:3d} | "
                f"BP={m['blocking_probability']:.4f} "
                f"frag={m['fragmentation']:.4f} "
                f"util={m['utilisation']:.4f} "
                f"QKD={m['qkd_success_rate']:.4f} "
                f"R={m['mean_reward']:.3f}"
            )

        # ---------------------------------------
        # Offline Training
        # ---------------------------------------
        for ep in range(1, OFFLINE_EPISODES + 1):

            print(
                f"Episode {ep} | "
                f"Load = {load:.1f} Erlangs"
            )

            m = run_episode(
                env,
                agent,
                episode_seed=ep,
                training=True
            )

            log(
                "offline",
                ep,
                m
            )

        # ---------------------------------------
        # Online Fine Tuning
        # ---------------------------------------
        for ep in range(1, ONLINE_EPISODES + 1):

            m = run_episode(
                env,
                agent,
                episode_seed=1000 + ep,
                training=True
            )

            log(
                "online",
                ep,
                m
            )

        # ---------------------------------------
        # Evaluation
        # ---------------------------------------
        eval_frag = []

        for ep in range(1, EVAL_EPISODES + 1):

            m = run_episode(
                env,
                agent,
                episode_seed=9000 + ep,
                training=False
            )

            log(
                "eval",
                ep,
                m
            )

            eval_frag.append(m["fragmentation"])

        # Mean external fragmentation over ALL evaluation
        # episodes for this load (not a single episode).
        avg_eval_frag = float(np.mean(eval_frag))

        eval_summary["load"].append(load)
        eval_summary["avg_eval_fragmentation"].append(
            avg_eval_frag
        )

        print(
            f"\n>>> Load {load} Erlangs | mean external "
            f"fragmentation over {EVAL_EPISODES} eval "
            f"episodes = {avg_eval_frag:.4f}"
        )

        agent.save(
            f"ppo_load_{load}"
        )

    pd.DataFrame(history).to_csv(
        "results.csv",
        index=False
    )

    pd.DataFrame(eval_summary).to_csv(
        "eval_fragmentation_summary.csv",
        index=False
    )

    print("\nTraining completed.")

    print(
        "Results saved to results.csv"
    )


if __name__ == "__main__":
    main()

