import numpy as np
import tensorflow as tf

NEG_INF = -1e9   # logit value for masked (infeasible) actions.


class PPOAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, lam=0.95, clip_ratio=0.2,
                 actor_lr=3e-4, critic_lr=1e-3,
                 train_iters=5, minibatch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self._buffer = []   # rollout: (state, action, logp, reward, value, mask)

    # ------------------------------------------------------------- networks
    def _build_actor(self):
        # Hidden-layer pattern follows DQN.py (relu stack -> logits head).
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        return model

    def _build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ])
        return model

    # ---------------------------------------------------- masked policy ----
    def _masked_logits(self, states, masks):
        logits = self.actor(states)
        masks = tf.cast(masks, tf.float32)
        return logits + (1.0 - masks) * NEG_INF

    def act(self, state, mask):
        """Sample a feasible action; return (action, log_prob, value)."""
        s = state[None, :]
        m = mask[None, :]
        logits = self._masked_logits(s, m)
        logp_all = tf.nn.log_softmax(logits)
        probs = tf.exp(logp_all)[0].numpy()
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            # No feasible action -> forced block handled by the environment.
            return int(np.argmax(mask)), 0.0, float(self.critic(s)[0, 0])
        probs = probs / total
        action = int(np.random.choice(self.action_size, p=probs))
        logp = float(logp_all[0, action].numpy())
        value = float(self.critic(s)[0, 0])
        return action, logp, value

    # ----------------------------------------------------- rollout buffer --
    def remember(self, state, action, logp, reward, value, mask):
        self._buffer.append((state, action, logp, reward, value, mask))

    def penalize_last(self, penalty):
        """Fold a blocking penalty into the most recent transition.

        Requests blocked with an all-zero action mask involve no
        decision, but their -1 reward is a consequence of the
        preceding allocation decisions that congested the spectrum.
        Adding it to the last stored transition lets GAE propagate
        the blocking signal back through the trajectory; without
        this, a fully masked environment would only ever train on
        +1 rewards.
        """
        if self._buffer:
            s, a, lp, r, v, m = self._buffer[-1]
            self._buffer[-1] = (s, a, lp, r + penalty, v, m)

    def _compute_gae(self, rewards, values, last_value):
        """Generalised Advantage Estimation."""
        adv = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        returns = adv + np.asarray(values, dtype=np.float32)
        return adv, returns

    # ------------------------------------------------ Section 10: PPO update
    ENTROPY_COEF = 0.01   # c2 in the canonical PPO objective

    def train(self, last_value=0.0):
        """PPO update with the full canonical objective
        L = L_CLIP + c2 * S[pi]  (Schulman et al., 2017),
        plus training-health diagnostics."""
        if not self._buffer:
            return None
        states = np.array([b[0] for b in self._buffer], dtype=np.float32)
        actions = np.array([b[1] for b in self._buffer], dtype=np.int32)
        old_logp = np.array([b[2] for b in self._buffer], dtype=np.float32)
        rewards = [b[3] for b in self._buffer]
        values = [b[4] for b in self._buffer]
        masks = np.array([b[5] for b in self._buffer], dtype=np.float32)

        adv, returns = self._compute_gae(rewards, values, last_value)

        # Critic loss (reference formula): MSE between predicted
        # state values and bootstrapped target values,
        #   Loss = (1/N) * sum (target_t - V(s_t))^2,
        # where the bootstrapped target is the lambda-return
        #   target_t = R_t + gamma * [(1-lambda) V(s_{t+1})
        #                             + lambda * target_{t+1}]
        # (= `returns` from GAE). The pure one-step TD(0) target
        # R_t + gamma*V(s_{t+1}) is the lambda=0 special case; it
        # propagates return information only one step per update,
        # which starves the critic over this ~100-step reward
        # horizon and stalls the actor.
        td_targets = returns

        raw_adv_std = float(adv.std())
        if raw_adv_std > 1e-6:
            adv = (adv - adv.mean()) / (raw_adv_std + 1e-8)
        # else: degenerate batch (all rewards identical);
        # normalising would only amplify noise.

        # Canonical PPO optimisation: train_iters epochs over the
        # rollout, each epoch a fresh shuffle divided into
        # mini-batches (Schulman et al., 2017).
        n = len(states)
        actor_losses, critic_losses = [], []
        for _ in range(self.train_iters):
            order = np.random.permutation(n)
            for s0 in range(0, n, self.minibatch_size):
                mb = order[s0:s0 + self.minibatch_size]
                mb_states = states[mb]
                mb_masks = masks[mb]
                mb_actions = actions[mb]
                mb_oldlp = old_logp[mb]
                mb_adv = adv[mb]
                mb_td = td_targets[mb]

                with tf.GradientTape() as tape:
                    logits = self._masked_logits(
                        mb_states, mb_masks)
                    logp_all = tf.nn.log_softmax(logits)
                    onehot = tf.one_hot(
                        mb_actions, self.action_size)
                    logp = tf.reduce_sum(
                        onehot * logp_all, axis=1)
                    ratio = tf.exp(logp - mb_oldlp)
                    clipped = tf.clip_by_value(
                        ratio,
                        1 - self.clip_ratio,
                        1 + self.clip_ratio)
                    surrogate = tf.reduce_mean(
                        tf.minimum(ratio * mb_adv,
                                   clipped * mb_adv))
                    probs = tf.exp(logp_all)
                    entropy = tf.reduce_mean(
                        -tf.reduce_sum(
                            probs * logp_all, axis=1))
                    actor_loss = -(surrogate
                                   + self.ENTROPY_COEF
                                   * entropy)
                grads = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                self.actor_opt.apply_gradients(
                    zip(grads, self.actor.trainable_variables))
                actor_losses.append(float(actor_loss))

                with tf.GradientTape() as tape:
                    v = tf.squeeze(
                        self.critic(mb_states), axis=1)
                    # mean squared TD error
                    critic_loss = tf.reduce_mean(
                        (mb_td - v) ** 2)
                grads = tape.gradient(
                    critic_loss,
                    self.critic.trainable_variables)
                self.critic_opt.apply_gradients(
                    zip(grads,
                        self.critic.trainable_variables))
                critic_losses.append(float(critic_loss))

        # post-update full-batch statistics for diagnostics
        logits = self._masked_logits(states, masks)
        logp_all = tf.nn.log_softmax(logits)
        onehot = tf.one_hot(actions, self.action_size)
        logp = tf.reduce_sum(onehot * logp_all, axis=1)
        ratio = tf.exp(logp - old_logp)
        probs = tf.exp(logp_all)
        entropy_v = float(tf.reduce_mean(
            -tf.reduce_sum(probs * logp_all, axis=1)))
        kl_v = float(tf.reduce_mean(old_logp - logp))
        clip_v = float(tf.reduce_mean(tf.cast(
            tf.abs(ratio - 1.0) > self.clip_ratio,
            tf.float32)))

        # ---- diagnostics (PPO health metrics) ----
        rew = np.asarray(rewards, dtype=np.float32)
        v_pred = np.asarray(values, dtype=np.float32)
        var_ret = float(np.var(returns))
        explained_var = (
            1.0 - float(np.var(returns - v_pred)) / var_ret
            if var_ret > 1e-8 else 0.0
        )
        diag = {
            "buffer_size": len(rew),
            "gradient_steps": len(actor_losses),
            "reward_min": float(rew.min()),
            "reward_mean": float(rew.mean()),
            "n_negative_rewards": int((rew < 0).sum()),
            "raw_adv_std": raw_adv_std,
            "explained_variance": explained_var,
            "policy_entropy": entropy_v,
            "approx_kl": kl_v,
            "clip_fraction": clip_v,
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
        }

        self._buffer.clear()
        return diag

    # ------------------------------------------------------- checkpointing -
    def save(self, prefix):
        self.actor.save(prefix + "_actor.keras")
        self.critic.save(prefix + "_critic.keras")

    def load(self, prefix):
        self.actor = tf.keras.models.load_model(prefix + "_actor.keras")
        self.critic = tf.keras.models.load_model(prefix + "_critic.keras")