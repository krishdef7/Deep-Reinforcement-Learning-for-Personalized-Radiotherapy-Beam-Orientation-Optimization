# 🧠 Deep Reinforcement Learning for Personalized Radiotherapy Beam Orientation

This repository contains the code and experiments for my work on **patient-specific Beam Orientation Optimization (BOO)** in head-and-neck radiotherapy using **Deep Q-Learning (DQN)**. The agent learns to select clinically meaningful gantry angles directly from voxel-level anatomy, **without** repeated Monte Carlo dose simulations.

> **TL;DR:** Given CT anatomy and organ masks, a DQN picks 5 beam angles in under a second. On 100 held-out OpenKBP head-and-neck patients it raises mean PTV coverage from **0.687 → 0.806** (+11.9 points) and D95 from **0.121 → 0.241** (~2×) vs standard equiangular plans. Trade-off: mean OAR dose is higher than equiangular (parotids, mandible; see Results), so OAR sparing is the main open problem.


---

## 🔗 Quick Navigation
- 🔍 Overview & Motivation
- 📈 Results (100 Patients)
- 📂 Repository Structure
- ⚙️ Installation
- ▶️ Evaluation
- 🏋️ Training
- 🧬 Model Summary
- 🔮 Future Work
- 📄 Citation
- 🙏 Acknowledgements


---

## 📌 Overview / Problem

Selecting clinically optimal beam orientations is crucial in radiotherapy.  
Conventional BOO methods:

-  Not personalized to anatomy ❌
-  Computationally infeasible at large search spaces ❌
-  Insensitive to voxel-level geometry ❌
-  Require repeated full dose simulations ❌

---

## 🚀 Core Idea

We formulate BOO as a **sequential decision-making problem** and train a Deep Q-Network to:

- Extract **voxel-level anatomical structure** from CT + organ masks
- Sequentially choose **5 distinct beam angles**
- Accumulate a **pseudo-physical dose surrogate** over timesteps
- Optimize reward balancing:
  - **PTV coverage** (good)
  - **OAR sparing** (avoid toxicity)

Inference time: **<1 second** per patient.


---

## 📁 Repository Structure

```
Deep-Reinforcement-Learning-for-Personalized-Radiotherapy-Beam-Orientation-Optimization/
├── configs/
│   └── experiments.json
├── figures/
│   ├── success_cases/      # Best examples
│   ├── typical_cases/      # Typical
│   ├── failure_cases/     # Failure cases
│   └── anomaly_cases/     # Special discussion
├── models/
│   └── best_dqn_model.pt
├── results/
│   ├── summary_results.md
│   └── test_results.csv
├── utils/
│   └── repro.py
├── baselines.py
├── eval_main.py
├── train.py
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation

```bash
git clone https://github.com/krishdef7/Deep-Reinforcement-Learning-for-Personalized-Radiotherapy-Beam-Orientation-Optimization.git
cd Deep-Reinforcement-Learning-for-Personalized-Radiotherapy-Beam-Orientation-Optimization
pip install -r requirements.txt
```


---

## 📂 Dataset (OpenKBP)

We use the OpenKBP dataset (head-and-neck):

- CT volumes  
- PTV mask  
- OAR masks (cord, brainstem, L/R parotids, mandible)

Split:

- Train: **200**
- Validation: **40**
- Test: **100**

🛠 Users must download OpenKBP separately and update paths in `configs/experiments.json`


---

## ▶️ Running Evaluation (Generate Results + Figures)

```bash
python eval_main.py
```

Outputs will include:

- `results/test_results.csv` — per-patient metrics (D95, coverage, OAR doses)
- `figures/patient_XXX_dose_dqn.png` — DQN dose maps overlaid on CT
- `figures/patient_XXX_dvh_dqn.png` — dose–volume histogram plots


---

## 🏋️ Training from Scratch

```bash
python train.py
```

Training summary:

- Replay buffer: 3000
- Batch size: 32
- γ = 0.95
- ε-greedy: 0.90 → 0.10
- Target network update every 5 epochs
- Converges in ~3.5 hours CPU


---

## 📈 Results — 100 Patient Evaluation

| Method        | Coverage  | D95     |
|--------------|-----------|---------|
| **DQN (ours)** | **0.8059** | **0.2405** |
| Equiangular   | 0.6867    | 0.1207 |
| Heuristic     | 0.6397    | 0.0949 |
| RandomMean    | 0.5883    | 0.0554 |

### Key Highlights
- **+11.9 points absolute** in mean PTV coverage (0.687 → 0.806)
- **~2× improvement** in D95
- **<1 second** per patient (post-training)
- Evaluated on **100 unseen CT cases**
- **Trade-off:** mean OAR dose is higher than equiangular. Left parotid 0.277 vs 0.191, right parotid 0.281 vs 0.276, mandible 0.233 vs 0.100 (normalized dose, `results/test_results.csv`). The agent buys target coverage partly at the OARs' expense.


---

## 🧬 Model Summary

**State (8 channels):**
- CT
- PTV mask
- 5 OAR masks
- Accumulating dose surrogate

**Actions:**
- 36 discrete gantry angles (0–350° at 10° spacing)
- DQN selects **5 sequential non-repeating beams**

**Architecture:**
- 5× Conv layers + BN + ReLU
- Bottleneck: 4×4×256
- Fully connected head
- Masking to prevent repeated beams
- Model parameters: ~3.4M

**Dose Surrogate:**
1) Ray-traced geometric field  
2) Gaussian blur → approximate scatter  
3) Accumulate dose per timestep  

**Reward:**
- Terminal reward based on:
  - ↑ D95 and coverage
  - ↓ mean OAR dose


---

## 🎨 Qualitative Examples

Located in:

```
figures/success_cases/
figures/typical_cases/
figures/failure_cases/
figures/anomaly_cases/
```

Success cases keep high-dose regions inside the PTV. Failure and anomaly cases show where dose spills into OARs.  
DVH curves reflect improved target coverage.


---

## 📊 Baselines Implemented

All evaluated under **identical surrogate dose** to ensure fair comparison:

- **Equiangular beams**
- **Geometry heuristic**
- **Random non-repeating beams** (mean)


---

## 🧭 Clinical Interpretation

- Higher D95 → higher local tumor control likelihood
- OAR avoidance reduces severe toxicity risk
- <1s runtime enables:
  - Adaptive planning
  - Online replanning
  - QA workflow assistive tools


---

## 🚧 Limitations (Honest Assessment)

- Surrogate dose ≠ true Monte Carlo dose
- Current version operates on **single 2D slice**
- Trained only on **head-and-neck geometry**
- Research prototype — **not clinically deployable**


---

## 🔮 Future Directions

- 3D DQN / U-Net encoders
- GPU-based Monte Carlo integration
- Learned neural surrogate physics
- Multi-objective RL (Pareto optimal)
- Online robustness against anatomical changes
- Multi-disease training (lung, pelvis, liver)


---

## 📄 Citation

If you use this repository, please cite:

**Deep Reinforcement Learning for Personalized Radiotherapy Beam Orientation Optimization.  
Krish Garg, IIT Roorkee, 2025.**


---

## 🙏 Acknowledgements

- OpenKBP dataset contributors
- IIT Roorkee — Department of Physics (institutional affiliation)
- No external funding used

