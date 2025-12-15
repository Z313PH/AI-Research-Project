import os, sys
SUMO_HOME = os.environ.get("SUMO_HOME")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci

CFG="scenarios/single_intersection/sim.sumocfg"
TLS="tls0"

traci.start(["sumo", "-c", CFG, "--start"])
logics = traci.trafficlight.getAllProgramLogics(TLS)
phases = logics[0].phases

links = traci.trafficlight.getControlledLinks(TLS)  
print("Num signal indices:", len(links))

for i,p in enumerate(phases):
    state = p.state
    lane_states = {}
    for idx,ch in enumerate(state):
        for (inLane, outLane, via) in links[idx]:
            lane_states.setdefault(inLane, []).append(ch)
    print(f"\nPHASE {i} state={state}")
    for lane, chars in sorted(lane_states.items()):
        # if any connection from that lane is green, treat lane as green
        is_green = any(c in ("G","g") for c in chars)
        print(f"  {lane}: {'GREEN' if is_green else 'RED'} ({''.join(chars)})")

traci.close()
