#  User Manual

**Project:** Single-intersection traffic signal control in SUMO using a CO₂-aware Deep Q-Network (DQN)

## 1) What this software does
This project trains and evaluates a reinforcement-learning traffic signal controller for a **single 4-way intersection** in **SUMO**.

Controllers provided:
- **Fixed-time**: Uses SUMO’s default traffic light program (no overrides).
- **Actuated baseline**: Simple demand/queue-based switching (detector-inspired).
- **CO₂-aware DQN**: Learns a policy that minimizes a CO₂-weighted cost while also penalizing delay, queueing, and excessive switching.

Outputs:
- Trained model checkpoint: `outputs/dqn_co2.pt`
- Evaluation logs: `outputs/eval_*.csv`

---

## 2) Requirements
### Software
- **SUMO** (includes TraCI tools)
- **Python 3**
- **PyTorch**

### Python packages
Install required Python packages (at minimum):
```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch
```

---

## 3) Project structure
Expected directory layout:
```
software/
  scenarios/single_intersection/
    net.net.xml
    routes.rou.xml
    tls.add.xml
    sim.sumocfg
    ...
  src/
    env_sumo.py
    controllers.py
    train_dqn.py
    eval.py
    smoke_test.py
    utils.py
  outputs/
```

---

## 4) SUMO setup (macOS / Homebrew)
### 4.1 Set SUMO_HOME (required)
SUMO’s TraCI Python modules live in `$SUMO_HOME/tools`.

```bash
export SUMO_HOME="$(brew --prefix sumo)/share/sumo"
export PYTHONPATH="$PYTHONPATH:$SUMO_HOME/tools"
```

To verify TraCI imports:
```bash
python3 -c "import os, sys; sys.path.append(os.path.join(os.environ['SUMO_HOME'],'tools')); import traci; print('traci OK')"
```

---

## 5) Quick smoke test (optional)
Runs a short simulation and prints basic metrics.
```bash
python3 src/smoke_test.py
```

---

## 6) Training the CO₂-aware DQN
From the project root (`software/`):

```bash
python3 src/train_dqn.py --reward co2 --episodes 200 --out outputs/dqn_co2.pt --seed 1000
```

Notes:
- Training uses an ε-greedy policy with a linear decay.
- Episodes use seeds `seed+ep` for reproducibility.

---

## 7) Evaluating controllers
All evaluations use a fixed horizon (default: 3600 seconds).

### 7.1 Fixed-time baseline
```bash
python3 src/eval.py --controller fixed --episodes 20 --seed 1000 --reward co2
```

### 7.2 Actuated baseline
```bash
python3 src/eval.py --controller actuated --episodes 20 --seed 1000 --reward co2
```

### 7.3 DQN controller
```bash
python3 src/eval.py --controller dqn --episodes 20 --seed 1000 --reward co2 --model outputs/dqn_co2.pt
```

---

## 8) Understanding the CSV output
Each evaluation writes a CSV like `outputs/eval_YYYYMMDD_HHMMSS.csv` with columns:
- `controller` (fixed_time / actuated / dqn)
- `episode`, `seed`, `steps`
- `co2_total`, `fuel_total`, `queue_sum`, `delay_sum`

Common interpretation:
- **CO₂** is accumulated from per-vehicle emissions over time. If logged in mg, convert to kg via: `kg = mg / 1e6`.
- **Delay sum** is cumulative incremental waiting time.
- **Queue sum** is cumulative halting vehicles aggregated over time.

---

## 9) Common issues & fixes
### 9.1 `ModuleNotFoundError: No module named 'traci'`
Cause: SUMO tools path not added.
Fix:
```bash
export SUMO_HOME="$(brew --prefix sumo)/share/sumo"
export PYTHONPATH="$PYTHONPATH:$SUMO_HOME/tools"
```

### 9.2 `FXApp::openDisplay: unable to open display :0.0`
Cause: `sumo-gui` requires a GUI display.
Fix: Use command-line SUMO:
```bash
sumo -c scenarios/single_intersection/sim.sumocfg
```

### 9.3 Lane ID errors (e.g., lane not known)
Cause: Configured lane IDs don’t match the network.
Fix: Ensure you are using the correct incoming lanes:
- `E2J_0, N2J_0, S2J_0, W2J_0`

---

## 10) Reproducibility notes
- Training and evaluation seeds are set explicitly.
- Evaluation runs multiple episodes (seeds 1000–1019) and reports mean ± std.

