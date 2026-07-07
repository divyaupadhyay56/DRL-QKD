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

# The PPO agent is trained ONCE, in a single continuous
# training phase at the fixed training load. The saved model
# is then frozen and evaluated across a range of loads.
TRAIN_LOAD = 300

EVAL_LOADS = [150, 200, 250, 300, 350, 400]

EPISODE_SIZE = 1500
NUM_TRAINING_SLOTS = 250

# Canonical PPO rollout: update every ROLLOUT_SIZE transitions
# (shuffled mini-batches inside the update), instead of one
# full-batch update per 1500-request episode.
ROLLOUT_SIZE = 512

TRAINING_EPISODES = 1000
EVAL_EPISODES = 500

MODEL_PREFIX = "ppo_trained"


def discounted_return(rewards, gamma):
    """G_0 via the recursive form G_t = R_{t+1} + gamma*G_{t+1}."""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return float(g)


def run_episode(env, agent, episode_seed, training=True):

    env.traffic.rng = np.random.default_rng(episode_seed)

    env.reset()

    blocked = 0
    served = 0

    diag = None

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
            if training:
                # blocking is a consequence of earlier
                # placements: route the -1 into the rollout
                agent.penalize_last(-1.0)
            continue

        state = env.build_state(req, routes)       # it gives information of Network and available free blocks to PPO.

        mask = env.action_mask(req, routes)  #It informs PPO which blocks are valid for this request and which ones cannot be selected at all.

        if mask.sum() == 0:
            blocked += 1
            block_reasons[
                env.mask_block_reason(req, routes)
            ] += 1
            rewards.append(-1.0)
            if training:
                # blocking is a consequence of earlier
                # placements: route the -1 into the rollout
                agent.penalize_last(-1.0)
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

            if len(agent._buffer) >= ROLLOUT_SIZE:
                # mid-episode update: the episode continues, so
                # bootstrap with the critic's estimate of the
                # current state instead of 0.
                diag = agent.train(last_value=value)

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
        # Final partial rollout: the episode terminates here, so
        # the return beyond it is zero -> bootstrap with 0.
        end_diag = agent.train(last_value=0.0)
        if end_diag is not None:
            diag = end_diag

    if diag is not None:
        print(
            f"    [diag] loss={diag['critic_loss']:.1f}"
        )

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

        # Discounted cumulative reward (Sutton & Barto):
        #   G_t = R_{t+1} + gamma * G_{t+1}
        # i.e. G_0 = sum_k gamma^k * R_{k+1}, computed with
        # the recursive form over the episode's requests,
        # using the agent's own discount rate gamma.
        "discounted_cumulative_reward":
            discounted_return(rewards, agent.gamma),

        

        "blocked_route": block_reasons["route"] / n,
        "blocked_qc": block_reasons["qc"] / n,
        "blocked_cc": block_reasons["cc"] / n,
        "blocked_dc": block_reasons["dc"] / n,
        "blocked_combination":
            block_reasons["combination"] / n,
    }


def make_env(load):
    """Environment + traffic generator for a given offered load."""
    net = MCFNetwork()
    traffic = TrafficGenerator(
        nodes=net.nodes,
        arrival_rate=load / HOLDING_MEAN,
        holding_mean=HOLDING_MEAN
    )
    return QKDEnvironment(
        net,
        traffic,
        k_paths=K_PATHS,
        total_requests=EPISODE_SIZE,
        num_training_slots=NUM_TRAINING_SLOTS,
        j_blocks=J_BLOCKS
    )


METRIC_KEYS = (
    "blocking_probability",
    "fragmentation",
    "utilisation",
    "qkd_success_rate",
    "discounted_cumulative_reward",
    "blocked_route",
    "blocked_qc",
    "blocked_cc",
    "blocked_dc",
    "blocked_combination"
)


def main():

    history = {"load": [], "phase": [], "episode": []}
    for key in METRIC_KEYS:
        history[key] = []

    def log(load, phase, ep, m):
        history["load"].append(load)
        history["phase"].append(phase)
        history["episode"].append(ep)
        for key in METRIC_KEYS:
            if key == "discounted_cumulative_reward" \
                    and phase == "eval":
                # reward is a training metric only; it is not
                # reported for the evaluation phase
                history[key].append(np.nan)
                continue
            history[key].append(m[key])
        line = (
            f"[{phase}] load {load:3d} ep {ep:3d} | "
            f"BP={m['blocking_probability']:.4f} "
            f"frag={m['fragmentation']:.4f} "
            f"util={m['utilisation']:.4f} "
            f"QKD={m['qkd_success_rate']:.4f} "
            
        )
        if phase != "eval":
            line += f" G={m['discounted_cumulative_reward']:.2f}"
        print(line)

    # =======================================================
    # 1) TRAIN ONCE: a single continuous training phase at
    #    the fixed training load.
    # =======================================================
    print("=" * 60)
    print(f"Training PPO (single continuous phase) at "
          f"load = {TRAIN_LOAD} Erlangs")
    print("=" * 60)

    env = make_env(TRAIN_LOAD)

    agent = PPOAgent(env.state_size, env.action_size)

    print("state_size =", env.state_size,
          "action_size =", env.action_size)

    for ep in range(1, TRAINING_EPISODES + 1):
        m = run_episode(env, agent,
                        episode_seed=ep,
                        training=True)
        log(TRAIN_LOAD, "training", ep, m)

    # =======================================================
    # 2) SAVE the trained model
    # =======================================================
    agent.save(MODEL_PREFIX)
    print(f"\nTrained model saved as '{MODEL_PREFIX}_*.keras'")

    # =======================================================
    # 3-5) EVALUATE the frozen policy on multiple loads:
    #      load the saved model, no retraining, no updates.
    # =======================================================
    eval_agent = PPOAgent(env.state_size, env.action_size)
    eval_agent.load(MODEL_PREFIX)

    summary = {
        "load": [],
        "avg_blocking_probability": [],
        "avg_fragmentation": [],
        "avg_utilisation": [],
        "avg_qkd_success_rate": [],
    }

    for load in EVAL_LOADS:

        print("\n" + "=" * 60)
        print(f"Evaluating frozen policy at load = {load} Erlangs")
        print("=" * 60)

        eval_env = make_env(load)

        results = []

        for ep in range(1, EVAL_EPISODES + 1):
            m = run_episode(eval_env, eval_agent,
                            episode_seed=9000 + ep,
                            training=False)
            log(load, "eval", ep, m)
            results.append(m)

        def avg(key):
            return float(np.mean([r[key] for r in results]))

        summary["load"].append(load)
        summary["avg_blocking_probability"].append(
            avg("blocking_probability"))
        summary["avg_fragmentation"].append(
            avg("fragmentation"))
        summary["avg_utilisation"].append(
            avg("utilisation"))
        summary["avg_qkd_success_rate"].append(
            avg("qkd_success_rate"))
        

        print(
            f">>> Load {load} Erlangs (mean over "
            f"{EVAL_EPISODES} eval episodes): "
            f"BP={summary['avg_blocking_probability'][-1]:.4f} "
            f"frag={summary['avg_fragmentation'][-1]:.4f} "
            f"util={summary['avg_utilisation'][-1]:.4f} "
            f"QKD={summary['avg_qkd_success_rate'][-1]:.4f}"
        )

    pd.DataFrame(history).to_csv("results.csv", index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("eval_summary.csv", index=False)

    print("\nTraining + evaluation completed.")
    print("Per-episode results saved to results.csv")
    print("Per-load evaluation averages saved to eval_summary.csv")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

    