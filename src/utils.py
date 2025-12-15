from __future__ import annotations

import os
import sys
from typing import List

SCENARIO_DIR = "scenarios/single_intersection"
SUMO_CFG = os.path.join(SCENARIO_DIR, "sim.sumocfg")

# SUMO IDs
TLS_ID = "tls0"

IN_LANES = ["E2J_0", "N2J_0", "S2J_0", "W2J_0"]
NS_LANES = ["N2J_0", "S2J_0"]
EW_LANES = ["E2J_0", "W2J_0"]

# Control cadence (seconds)
DECISION_INTERVAL = 5


PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3


def add_sumo_tools_to_path() -> None:

    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME not set. Run:\n"
            "  export SUMO_HOME=\"$(brew --prefix sumo)/share/sumo\""
        )
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
