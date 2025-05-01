import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from env import QuadrupedEnv  # your custom env

# ————————————————————————————————————————————————————————————
#  Replay Buffer
# ————————————————————————————————————————————————————————————
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state_buf  = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.next_buf   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1),          dtype=np.float32)
        self.done_buf   = np.zeros((capacity, 1),          dtype=np.float32)

    def store(self, s, a, r, s2, d):
        self.state_buf[self.ptr]  = s
        self.action_buf[self.ptr] = a
        self.reward_buf[self.ptr] = r
        self.next_buf[self.ptr]   = s2
        self.done_buf[self.ptr]   = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state_buf[idxs]).to(device),
            torch.from_numpy(self.action_buf[idxs]).to(device),
            torch.from_numpy(self.reward_buf[idxs]).to(device),
            torch.from_numpy(self.next_buf[idxs]).to(device),
            torch.from_numpy(self.done_buf[idxs]).to(device),
        )

# ————————————————————————————————————————————————————————————
#  Networks
# ————————————————————————————————————————————————————————————
LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256), action_limit=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden[1], act_dim)
        self.log_std = nn.Linear(hidden[1], act_dim)
        self.action_limit = action_limit

    def forward(self, x):
        h = self.net(x)
        mean    = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        # reparameterization trick
        eps = torch.randn_like(mean)
        y = mean + eps * std
        action = torch.tanh(y) * self.action_limit

        # compute log prob
        log_prob = (
            -0.5 * ((y - mean) / (std + 1e-6)).pow(2)
            - 0.5 * np.log(2 * np.pi)
            - log_std
        ).sum(dim=-1, keepdim=True)
        # correction for Tanh
        log_prob -= (2*(np.log(2) - y - F.softplus(-2*y))).sum(dim=-1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

# ————————————————————————————————————————————————————————————
#  SAC Agent
# ————————————————————————————————————————————————————————————
class SACAgent:
    def __init__(self, obs_dim, act_dim, action_limit):
        # networks
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, action_limit=action_limit).to(device)

        # copy params to target
        for tgt, src in zip(self.q1_target.parameters(), self.q1.parameters()):
            tgt.data.copy_(src.data)
        for tgt, src in zip(self.q2_target.parameters(), self.q2.parameters()):
            tgt.data.copy_(src.data)

        # optimizers
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=3e-4)

        # entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        # hyperparams
        self.gamma = 0.99
        self.tau   = 0.005

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch):
        s, a, r, s2, d = batch

        # --- Critic loss ---
        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)
            q1_targ = self.q1_target(s2, a2)
            q2_targ = self.q2_target(s2, a2)
            q_targ = torch.min(q1_targ, q2_targ) - self.alpha * logp2
            backup = r + self.gamma * (1 - d) * q_targ

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, backup)
        q2_loss = F.mse_loss(q2_pred, backup)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # --- Policy loss ---
        a_new, logp_new = self.policy.sample(s)
        q1_new = self.q1(s, a_new)
        policy_loss = (self.alpha * logp_new - q1_new).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # --- Entropy (alpha) loss ---
        alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft update targets ---
        for tgt, src in zip(self.q1_target.parameters(), self.q1.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)
        for tgt, src in zip(self.q2_target.parameters(), self.q2.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)

# ————————————————————————————————————————————————————————————
#  Training loop
# ————————————————————————————————————————————————————————————
if __name__ == "__main__":
    # ─── Argument Parsing ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train SAC on the quadruped environment"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="quad_ex.xml",
        help="Path to your MuJoCo .xml model file"
    )
    args = parser.parse_args()

    # ─── Device ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Environment ────────────────────────────────────────────────────────────
    env = QuadrupedEnv(
        args.model_path,
        render_mode="human" if args.render else None
    )
    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    # ─── Agent & Replay Buffer ──────────────────────────────────────────────────
    agent = SACAgent(obs_dim, act_dim, act_limit).to(device)
    rb    = ReplayBuffer(obs_dim, act_dim)

    # ─── Training Hyperparameters ──────────────────────────────────────────────
    total_steps  = 200_000
    start_steps  = 5_000    # random policy warm-up
    update_after = 1_000    # start updates after this many env steps
    update_every = 50       # frequency of gradient updates
    batch_size   = 256

    # ─── Main Loop ─────────────────────────────────────────────────────────────
    state, _ = env.reset()
    episode_return = 0.0

    for t in range(1, total_steps + 1):
        # pick action
        if t < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                action, _ = agent.policy.sample(s_tensor)
                action = action.cpu().numpy()[0]

        # step
        next_s, reward, done, truncated, _ = env.step(action)
        rb.store(state, action, reward, next_s, float(done or truncated))

        state = next_s
        episode_return += reward

        # optionally render
        if args.render:
            env.render()

        # episode end
        if done or truncated:
            print(f"Step {t:>6} | Return: {episode_return:.2f}")
            state, _ = env.reset()
            episode_return = 0.0

        # updates
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = rb.sample_batch(batch_size)
                agent.update(batch)

    # ─── Save ───────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    torch.save({
        "q1":        agent.q1.state_dict(),
        "q2":        agent.q2.state_dict(),
        "q1_targ":   agent.q1_target.state_dict(),
        "q2_targ":   agent.q2_target.state_dict(),
        "policy":    agent.policy.state_dict(),
        "log_alpha": agent.log_alpha,
    }, "models/sac_tesbot.pth")

    print("Training complete!")
    env.close()
