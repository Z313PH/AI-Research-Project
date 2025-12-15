from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from utils import (
    add_sumo_tools_to_path,
    SUMO_CFG,
    IN_LANES,
    TLS_ID,
    DECISION_INTERVAL,
    PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_EW_GREEN, PHASE_EW_YELLOW,
)

add_sumo_tools_to_path()

import traci

@dataclass
class EnvConfig:
    sumocfg_path: str
    tls_id: str
    controlled_lanes: Tuple[str, ...]

    decision_interval_s: int = 5
    yellow_duration_s: int = 3
    min_green_s: int = 10
    seed: int = 0
    quiet: bool = True

    end_time_s: int = 3600

    normalize_state: bool = True
    norm_queue: float = 20.0
    norm_count: float = 20.0
    norm_tss: float = 60.0

    w_co2_kg: float = 1.0
    w_delay: float = 0.01
    w_queue: float = 0.05
    w_switch: float = 2.0


class SumoIntersectionEnv:

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._time_since_dir_change = 0.0
        self._last_dir = 0
        self._veh_wait_prev: Dict[str, float] = {}


    def _current_phase(self) -> int:
        return int(traci.trafficlight.getPhase(self.cfg.tls_id))

    def _current_dir(self) -> int:
        ph = self._current_phase()
        if ph in (PHASE_NS_GREEN, PHASE_NS_YELLOW):
            return 0
        return 1

    def _dir_to_green_phase(self, d: int) -> int:
        return PHASE_NS_GREEN if d == 0 else PHASE_EW_GREEN

    def _dir_to_yellow_phase(self, d: int) -> int:
        return PHASE_NS_YELLOW if d == 0 else PHASE_EW_YELLOW

    def _lane_queue(self, lane_id: str) -> float:
        return float(traci.lane.getLastStepHaltingNumber(lane_id))

    def _lane_count(self, lane_id: str) -> float:
        return float(traci.lane.getLastStepVehicleNumber(lane_id))

    def _step_emissions(self):
        # CO2 in mg/s, fuel in ml/s 
        vids = traci.vehicle.getIDList()
        if not vids:
            return 0.0, 0.0
        co2 = 0.0
        fuel = 0.0
        for vid in vids:
            co2 += float(traci.vehicle.getCO2Emission(vid))
            fuel += float(traci.vehicle.getFuelConsumption(vid))
        return co2, fuel

    def _step_delay(self) -> float:
        vids = traci.vehicle.getIDList()
        if not vids:
            self._veh_wait_prev.clear()
            return 0.0

        delay_inc = 0.0
        new_prev: Dict[str, float] = {}
        for vid in vids:
            try:
                wt = float(traci.vehicle.getAccumulatedWaitingTime(vid))
            except Exception:
                wt = float(traci.vehicle.getWaitingTime(vid))
            prev = self._veh_wait_prev.get(vid, wt)
            delay_inc += max(0.0, wt - prev)
            new_prev[vid] = wt
        self._veh_wait_prev = new_prev
        return delay_inc

    def _update_dir_timer(self):
        d = self._current_dir()
        if d != self._last_dir:
            self._last_dir = d
            self._time_since_dir_change = 0.0
        else:
            self._time_since_dir_change += 1.0

    def _simulate_seconds(self, seconds: int):
        co2_sum = fuel_sum = queue_sum = delay_sum = 0.0
        for _ in range(max(0, int(seconds))):
            traci.simulationStep()
            self._update_dir_timer()
            co2, fuel = self._step_emissions()
            delay = self._step_delay()
            queue = sum(self._lane_queue(l) for l in self.cfg.controlled_lanes)
            co2_sum += co2
            fuel_sum += fuel
            delay_sum += delay
            queue_sum += queue
        return co2_sum, fuel_sum, queue_sum, delay_sum

    def reset(self) -> np.ndarray:
        if traci.isLoaded():
            traci.close(False)

        cmd = ["sumo", "-c", self.cfg.sumocfg_path, "--start", "--quit-on-end", "--seed", str(int(self.cfg.seed))]
        if self.cfg.quiet:
            cmd += ["--no-step-log", "true", "--duration-log.disable", "true"]
        traci.start(cmd)

        lane_ids = set(traci.lane.getIDList())
        missing = [l for l in self.cfg.controlled_lanes if l not in lane_ids]
        if missing:
            traci.close(False)
            raise RuntimeError(
                f"Controlled lanes not found in network: {missing}. "
                f"Available lanes include: {sorted(list(lane_ids))[:12]} ..."
            )

        self._veh_wait_prev.clear()
        self._last_dir = self._current_dir()
        self._time_since_dir_change = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        queues = [self._lane_queue(l) for l in self.cfg.controlled_lanes]
        counts = [self._lane_count(l) for l in self.cfg.controlled_lanes]
        cur_dir = float(self._current_dir())
        tss = float(self._time_since_dir_change)

        if self.cfg.normalize_state:
            queues = [q / self.cfg.norm_queue for q in queues]
            counts = [c / self.cfg.norm_count for c in counts]
            tss = tss / self.cfg.norm_tss

        return np.array(queues + counts + [cur_dir, tss], dtype=np.float32)


    def step(self, action: Optional[int], reward_mode: str = "co2"):
        ph = self._current_phase()
        if ph in (PHASE_NS_YELLOW, PHASE_EW_YELLOW):
            action = None

        switch_penalty = 0.0

        if action is not None:
            action = int(action)
            if action not in (0, 1):
                raise ValueError("Action must be 0 (NS) or 1 (EW) or None")

            current_dir = self._current_dir()
            if action != current_dir and self._time_since_dir_change >= self.cfg.min_green_s:
                traci.trafficlight.setPhase(self.cfg.tls_id, self._dir_to_yellow_phase(current_dir))
                self._simulate_seconds(self.cfg.yellow_duration_s)
                traci.trafficlight.setPhase(self.cfg.tls_id, self._dir_to_green_phase(action))
                switch_penalty = 1.0

        if switch_penalty:
            remaining = max(0, int(self.cfg.decision_interval_s) - int(self.cfg.yellow_duration_s))
            co2_sum, fuel_sum, queue_sum, delay_sum = self._simulate_seconds(remaining)
        else:
            co2_sum, fuel_sum, queue_sum, delay_sum = self._simulate_seconds(self.cfg.decision_interval_s)

        sim_t = float(traci.simulation.getTime())
        done = sim_t >= float(self.cfg.end_time_s)
        
        co2_kg = co2_sum / 1e6

        if reward_mode == "co2":
            reward = -(
                self.cfg.w_co2_kg * co2_kg
                + self.cfg.w_delay * delay_sum
                + self.cfg.w_queue * queue_sum
                + self.cfg.w_switch * switch_penalty
            )
        else:
            reward = -(
                self.cfg.w_delay * delay_sum
                + self.cfg.w_queue * queue_sum
                + 0.1 * self.cfg.w_co2_kg * co2_kg
                + self.cfg.w_switch * switch_penalty
            )

        info = {
            "co2": co2_sum,
            "fuel": fuel_sum,
            "queue": queue_sum,
            "delay": delay_sum,
            "switch": switch_penalty,
            "phase": self._current_phase(),
            "dir": self._current_dir(),
            "seed": int(self.cfg.seed),
        }
        return self._get_state(), float(reward), bool(done), info

    def close(self):
        if traci.isLoaded():
            traci.close(False)
