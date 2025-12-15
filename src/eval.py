from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import List, Optional

from env_sumo import EnvConfig, SumoIntersectionEnv
from controllers import FixedTimeController, SimpleActuatedController
from utils import SUMO_CFG, TLS_ID, NS_LANES, EW_LANES, IN_LANES


def run_controller(
    controller_name: str,
    controller,
    episodes: int,
    reward_mode: str,
    model_path: Optional[str] = None,
    base_seed: int = 1000,
) -> List[dict]:
    cfg = EnvConfig(
        sumocfg_path=SUMO_CFG,
        tls_id=TLS_ID,
        controlled_lanes=tuple(IN_LANES),
        seed=base_seed,
        end_time_s=3600,
        normalize_state=True,
    )
    env = SumoIntersectionEnv(cfg)

    qnet = None
    if controller_name.startswith("dqn"):
        import torch 
        from train_dqn import QNet


        s0 = env.reset()
        state_dim = int(s0.shape[0])
        env.close()

        qnet = QNet(state_dim, 2)
        qnet.load_state_dict(torch.load(model_path, map_location="cpu"))
        qnet.eval()


        env = SumoIntersectionEnv(cfg)

    rows: List[dict] = []
    for ep in range(episodes):
        cfg.seed = base_seed + ep
        s = env.reset()
        done = False
        ep_co2 = ep_fuel = ep_queue = ep_delay = ep_reward = 0.0
        steps = 0

        while not done:
            if qnet is not None:
                import torch
                with torch.no_grad():
                    a = int(torch.argmax(qnet(torch.tensor(s).float().unsqueeze(0)), dim=1).item())
            else:
                a = controller.act(s)  

            s, r, done, info = env.step(a, reward_mode=reward_mode)
            ep_reward += float(r)
            ep_co2 += float(info["co2"])
            ep_fuel += float(info["fuel"])
            ep_queue += float(info["queue"])
            ep_delay += float(info["delay"])
            steps += 1

        rows.append({
            "controller": controller_name,
            "episode": ep,
            "seed": base_seed + ep,
            "steps": steps,
            "reward": ep_reward,
            "co2_total": ep_co2,
            "fuel_total": ep_fuel,
            "queue_sum": ep_queue,
            "delay_sum": ep_delay,
            "reward_mode": reward_mode,
        })

    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["fixed", "actuated", "dqn"], default="fixed")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--reward", choices=["co2", "standard"], default="co2")
    ap.add_argument(
        "--model",
        type=str,
        default="outputs/dqn_co2.pt",
        help="Path to a trained DQN model (for --controller dqn)",
    )
    ap.add_argument("--seed", type=int, default=1000, help="Base SUMO seed (episode uses seed+ep)")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: outputs/eval_<timestamp>.csv)")
    args = ap.parse_args()

    if args.controller == "fixed":
        controller = FixedTimeController()  
        name = "fixed_time"
        rows = run_controller(name, controller, args.episodes, args.reward, base_seed=args.seed)

    elif args.controller == "actuated":
        controller = SimpleActuatedController(
            ns_lanes=NS_LANES,
            ew_lanes=EW_LANES,
            queue_margin=0.1,
        )
        name = "actuated"
        rows = run_controller(name, controller, args.episodes, args.reward, base_seed=args.seed)

    else:
        name = "dqn"
        rows = run_controller(
            "dqn",
            controller=None,
            episodes=args.episodes,
            reward_mode=args.reward,
            model_path=args.model,
            base_seed=args.seed,
        )

    os.makedirs("outputs", exist_ok=True)
    out_path = args.out or f"outputs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    avg_co2 = sum(r["co2_total"] for r in rows) / len(rows)
    avg_delay = sum(r["delay_sum"] for r in rows) / len(rows)
    print(f"[{name}] episodes={len(rows)} avg_CO2={avg_co2:.2f} mg ({avg_co2/1e6:.3f} kg) avg_delay={avg_delay:.2f} s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
