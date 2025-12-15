import os, sys
import numpy as np
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../software
SUMO_CFG = str(PROJECT_ROOT / "scenarios" / "single_intersection" / "sim.sumocfg")


SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    raise RuntimeError('SUMO_HOME not set. Run: export SUMO_HOME="$(brew --prefix sumo)/share/sumo"')
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

SUMO_CFG = "scenarios/single_intersection/sim.sumocfg"
TLS_ID = "tls0"

CONTROLLED_LANES = ["N2J_0","S2J_0","E2J_0","W2J_0"]

def lane_queue(lane):
    return traci.lane.getLastStepHaltingNumber(lane)

def lane_wait(lane):
    return traci.lane.getWaitingTime(lane)

def step_metrics():
    veh_ids = traci.vehicle.getIDList()

    co2 = 0.0
    fuel = 0.0
    for vid in veh_ids:
        co2 += traci.vehicle.getCO2Emission(vid)       # mg/s (per simulation step)
        fuel += traci.vehicle.getFuelConsumption(vid)  # ml/s 

    queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in CONTROLLED_LANES)
    delay = sum(traci.lane.getWaitingTime(l) for l in CONTROLLED_LANES)

    return float(co2), float(fuel), float(queue), float(delay)

    co2 = traci.simulation.getCO2Emission()
    fuel = traci.simulation.getFuelConsumption()
    queue = sum(lane_queue(l) for l in CONTROLLED_LANES)
    delay = sum(lane_wait(l) for l in CONTROLLED_LANES)
    return float(co2), float(fuel), float(queue), float(delay)

def main():
    traci.start(["sumo", "-c", SUMO_CFG, "--start"])
    logics = traci.trafficlight.getAllProgramLogics(TLS_ID)
    phases = logics[0].phases

    print("Phase states:")
    for i, p in enumerate(phases):
        print(i, p.state, p.duration)

    greenA = 0
    greenB = 2

    T = 300  # seconds total (5 minutes)
    decision_interval = 5
    t = 0

    co2_sum = fuel_sum = queue_sum = delay_sum = 0.0

    next_switch = 30
    current_green = greenA
    traci.trafficlight.setPhase(TLS_ID, current_green)

    while t < T:
        if t >= next_switch:
            current_green = greenB if current_green == greenA else greenA
            traci.trafficlight.setPhase(TLS_ID, current_green)
            next_switch += 30

        # step for decision_interval seconds
        for _ in range(decision_interval):
            traci.simulationStep()
            t += 1
            co2, fuel, q, d = step_metrics()
            co2_sum += co2
            fuel_sum += fuel
            queue_sum += q
            delay_sum += d

            if traci.simulation.getMinExpectedNumber() <= 0:
                t = T
                break
    out = Path(PROJECT_ROOT) / "results.csv"
    header = ["controller","minutes","co2_total_mg","fuel_total_ml","queue_sum","delay_sum"]
    row = ["fixed_time_30s_toggle", 5, co2_sum, fuel_sum, queue_sum, delay_sum]

    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"\nSaved -> {out}")

    traci.close()
    print("\nSMOKE TEST RESULTS (5 min)")
    print("CO2 total:", co2_sum)
    print("Fuel total:", fuel_sum)
    print("Queue sum:", queue_sum)
    print("Delay sum:", delay_sum)

if __name__ == "__main__":
    main()
