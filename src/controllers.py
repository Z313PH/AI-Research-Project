from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import traci


class FixedTimeController:

    def act(self, state) -> Optional[int]:
        return None


@dataclass
class SimpleActuatedController:

    ns_lanes: Sequence[str]
    ew_lanes: Sequence[str]
    queue_margin: float = 0.10   
    min_green_s: int = 10      

    def act(self, state) -> int:
        qE, qN, qS, qW = state[0], state[1], state[2], state[3]
        cur_dir = int(round(float(state[-2])))  
        tss = float(state[-1])   

        if tss < (self.min_green_s / 60.0):   
            return cur_dir

        qNS = float(qN + qS)
        qEW = float(qE + qW)

        if qNS > qEW + self.queue_margin:
            return 0
        if qEW > qNS + self.queue_margin:
            return 1

        return cur_dir