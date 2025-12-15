from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Deque, Tuple

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_sumo import EnvConfig, SumoIntersectionEnv
from utils import SUMO_CFG, TLS_ID, IN_LANES


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainConfig:
    episodes: int = 60
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    start_learn: int = 1_000
    target_sync: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 80_000
    max_steps_per_ep: int = 2_000
    seed: int = 0


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / float(cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["co2", "standard"], default="co2")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000, help="Base SUMO seed (episode uses seed+ep)")
    ap.add_argument("--out", type=str, default="outputs/dqn_co2.pt")
    args = ap.parse_args()

    tcfg = TrainConfig(episodes=args.episodes)
    random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    env_cfg = EnvConfig(
        sumocfg_path=SUMO_CFG,
        tls_id=TLS_ID,
        controlled_lanes=tuple(IN_LANES),
        seed=args.seed,
        end_time_s=3600,
        normalize_state=True,
    )    
    env = SumoIntersectionEnv(env_cfg)

    # Infer state dimension
    s0 = env.reset()
    state_dim = int(s0.shape[0])
    n_actions = 2
    env.close()

    q = QNet(state_dim, n_actions)
    q_tgt = QNet(state_dim, n_actions)
    q_tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=tcfg.lr)
    loss_fn = nn.SmoothL1Loss()

    replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=tcfg.replay_size)

    global_step = 0
    os.makedirs("outputs", exist_ok=True)

    for ep in range(tcfg.episodes):
        env.cfg.seed = args.seed + ep
        s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < tcfg.max_steps_per_ep:
            eps = epsilon_by_step(global_step, tcfg)
            if random.random() < eps:
                a = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.tensor(s).float().unsqueeze(0)), dim=1).item())

            s2, r, done, info = env.step(a, reward_mode=args.reward)
            replay.append((s, a, r, s2, done))
            s = s2
            ep_reward += r
            steps += 1
            global_step += 1

            if len(replay) >= tcfg.start_learn:
                batch = random.sample(replay, tcfg.batch_size)
                sb = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
                ab = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
                rb = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
                s2b = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
                db = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)

                qsa = q(sb).gather(1, ab)
                with torch.no_grad():
                    q_next = torch.max(q_tgt(s2b), dim=1, keepdim=True)[0]
                    target = rb + (1.0 - db) * tcfg.gamma * q_next

                loss = loss_fn(qsa, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if global_step % tcfg.target_sync == 0:
                q_tgt.load_state_dict(q.state_dict())

        print(f"[ep {ep:03d}] seed={args.seed+ep} steps={steps:4d} reward={ep_reward:.2f}")

        if (ep + 1) % 10 == 0:
            torch.save(q.state_dict(), args.out)
            print(f"Saved model: {args.out}")

    torch.save(q.state_dict(), args.out)
    print(f"Final model saved: {args.out}")
    env.close()


if __name__ == "__main__":
    main()
