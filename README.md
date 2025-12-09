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
