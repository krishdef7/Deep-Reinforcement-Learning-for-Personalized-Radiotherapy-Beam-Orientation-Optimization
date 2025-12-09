# 🧠 Deep Reinforcement Learning for Personalized Radiotherapy Beam Orientation

This repository contains the code and experiments for our work on **patient-specific Beam Orientation Optimization (BOO)** in head-and-neck radiotherapy using **Deep Q-Learning**. The agent selects clinically meaningful gantry angles directly from voxel-level anatomy, **without** repeated Monte Carlo dose simulation.

> **TL;DR:** Given a CT + contours, this project picks 5 beam angles in <1s that significantly improve PTV coverage vs. equiangular baselines.

**Quick links**

- 🔍 [Problem & approach](#-overview--problem)
- 🧪 [Results (100-patient test set)](#-results--100-patient-evaluation)
- ⚙️ [How to run evaluation](#-reproducing-results)
- 🏋️ [How to train from scratch](#-training-from-scratch)
- 📁 [Repository structure](#-repository-structure)
- 📄 [Paper / citation](#-citation)

## 📌 Overview / Problem

Choosing good beam orientations is critical for high-quality radiotherapy plans.  
Conventional BOO strategies (equiangular templates, simple heuristics, combinatorial solvers):

- ❌ Are **not personalized** to anatomy
- ❌ Become **computationally infeasible** at scale
- ❌ Ignore **voxel-level geometry**
- ❌ Often require repeated, slow **dose calculations**

## 🚀 Proposed Solution

We formulate BOO as a **sequential decision problem** and train a Deep Q-Network (DQN) to:

- Read multi-channel 2D slices: **CT + PTV + 5 OAR masks + evolving dose**
- Select **5 non-repeating gantry angles** from 36 candidates (0–350° at 10° resolution)
- Accumulate a **pseudo-physical dose surrogate** over time
- Balance **PTV coverage** and **OAR avoidance** via a clinically-motivated reward

The system produces **patient-adaptive beam sets in < 1 second** (CPU only).

