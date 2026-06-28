"""
ppo_agent.py
============
PPO agent for the QKD-enabled SS-EON allocation environment.

Implements Section 10 of implementation_v2:
  - Algorithm: PPO (chosen over DeepRMSA's A3C). The clipped surrogate
    objective provides training stability, so DeepRMSA's window-based (FLX)
    mechanism is not used.
  - Designed for the offline-then-online schedule driven from main.py
    (pre-train on the simulator, then fine-tune online).

Built with TensorFlow/Keras to stay consistent with the user's DQN.py
(tf.keras Sequential models, Adam optimiser, .save() checkpoints). Action
masking from the environment is applied to the policy logits so every executed
action is physically secure (Section 9).

References (uploaded by the user):
  - implementation_v2.docx ....... spec (Section 10).
  - DQN.py ....................... user's tf.keras agent (network/optimiser style).
  - DeepRMSA_Agent.py (github.com/xiaoliangchenUCD/DeepRMSA) .. A3C agent the
                                   PPO design is contrasted against.
  - DeepRMSA_Aorks_2.pdf ......... DeepRMSA paper (DRL-for-RMSA reference).
"""

import numpy as np
import tensorflow as tf

NEG_INF = -1e9   # logit value for masked (infeasible) actions.


class PPOAgent:
    def __init__(self, state_size, action_size,
                 gamma=0.99, lam=0.95, clip_ratio=0.2,
                 actor_lr=3e-4, critic_lr=1e-3,
                 train_iters=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters

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
    def train(self, last_value=0.0):
        if not self._buffer:
            return None
        states = np.array([b[0] for b in self._buffer], dtype=np.float32)
        actions = np.array([b[1] for b in self._buffer], dtype=np.int32)
        old_logp = np.array([b[2] for b in self._buffer], dtype=np.float32)
        rewards = [b[3] for b in self._buffer]
        values = [b[4] for b in self._buffer]
        masks = np.array([b[5] for b in self._buffer], dtype=np.float32)

        adv, returns = self._compute_gae(rewards, values, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(self.action_size)
        actor_losses, critic_losses = [], []
        for _ in range(self.train_iters):
            with tf.GradientTape() as tape:
                logits = self._masked_logits(states, masks)
                logp_all = tf.nn.log_softmax(logits)
                onehot = tf.one_hot(actions, self.action_size)
                logp = tf.reduce_sum(onehot * logp_all, axis=1)
                ratio = tf.exp(logp - old_logp)
                clipped = tf.clip_by_value(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(
                    tf.minimum(ratio * adv, clipped * adv))
            grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(
                zip(grads, self.actor.trainable_variables))
            actor_losses.append(float(actor_loss))

            with tf.GradientTape() as tape:
                v = tf.squeeze(self.critic(states), axis=1)
                critic_loss = tf.reduce_mean((returns - v) ** 2)
            grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(
                zip(grads, self.critic.trainable_variables))
            critic_losses.append(float(critic_loss))

        self._buffer.clear()
        return np.mean(actor_losses), np.mean(critic_losses)

    # ------------------------------------------------------- checkpointing -
    def save(self, prefix):
        self.actor.save(prefix + "_actor.keras")
        self.critic.save(prefix + "_critic.keras")

    def load(self, prefix):
        self.actor = tf.keras.models.load_model(prefix + "_actor.keras")
        self.critic = tf.keras.models.load_model(prefix + "_critic.keras")
