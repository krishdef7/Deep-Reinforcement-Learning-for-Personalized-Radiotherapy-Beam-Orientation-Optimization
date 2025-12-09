# 📈 Evaluation Summary

### Dataset
- 100 held-out test cases (OpenKBP head-and-neck)
- Inputs: CT + PTV + 5 OAR structures
- Metrics:
  - **D95** (dose received by 95% of the PTV)
  - **PTV Coverage** (fraction of PTV receiving ≥ threshold dose)
  - **Mean OAR dose**

---

## 🧪 Mean Performance (100 Patients)

| Method         | Coverage | D95   |
|----------------|----------|-------|
| **DQN (ours)** | **0.8058** | **0.2405** |
| Equiangular    | 0.6867   | 0.1207 |
| Heuristic      | 0.6397   | 0.0949 |
| RandomMean     | 0.5883   | 0.0554 |

**Observation:**  
The DQN generally improves PTV coverage and D95 relative to equiangular and heuristic baselines, with the random-beam baseline performing worst on average.

---

# Summary of Representative Evaluation Cases (n = 20)

We highlight 20 representative test patients for which qualitative dose and DVH figures were generated:

**Patients:** 4, 25, 28, 29, 32, 40, 45, 49, 50, 56, 57, 60, 74, 80, 81, 83, 84, 88, 89, 90  

For each case, we compare DQN against the equiangular baseline in terms of **PTV coverage** and **D95**.

---

## 📊 Per-Patient Comparison (DQN vs Equiangular)

| Patient | Cov(DQN) | Cov(Eq) | D95(DQN) | D95(Eq) | DQN vs Eq (PTV)                    |
|--------|----------|---------|----------|---------|------------------------------------|
| 4      | 0.455    | 0.327   | 0.000    | 0.000   | ↑ Coverage, same D95               |
| 25     | 0.914    | 0.765   | 0.271    | 0.135   | ↑ Coverage, ↑ D95                  |
| 28     | 0.635    | 0.451   | 0.0003   | 0.0010  | ↑ Coverage, ↓ D95 (both very low)  |
| 29     | 0.616    | 0.403   | 0.0022   | ~0.000  | ↑ Coverage, ↑ D95                  |
| 32     | 0.673    | 0.536   | 0.0002   | 0.0054  | ↑ Coverage, ↓ D95                  |
| 40     | 0.916    | 0.736   | 0.379    | 0.052   | ↑ Coverage, ↑ D95                  |
| 45     | 0.945    | 0.807   | 0.492    | 0.160   | ↑ Coverage, ↑ D95                  |
| 49     | 0.523    | 0.580   | 0.0013   | 0.0094  | ↓ Coverage, ↓ D95                  |
| 50     | 0.866    | 0.801   | 0.117    | 0.145   | ↑ Coverage, slightly ↓ D95         |
| 56     | 0.839    | 0.769   | 0.178    | 0.105   | ↑ Coverage, ↑ D95                  |
| 57     | 0.755    | 0.642   | 0.083    | 0.045   | ↑ Coverage, ↑ D95                  |
| 60     | 0.852    | 0.676   | 0.170    | 0.023   | ↑ Coverage, ↑ D95                  |
| 74     | 0.994    | 0.894   | 0.902    | 0.310   | ↑ Coverage, ↑ D95 (near-ideal)     |
| 80     | 0.978    | 0.901   | 0.669    | 0.335   | ↑ Coverage, ↑ D95                  |
| 81     | 0.823    | 0.516   | 0.167    | 0.0036  | ↑ Coverage, ↑ D95                  |
| 83     | 0.949    | 0.751   | 0.504    | 0.044   | ↑ Coverage, ↑ D95                  |
| 84     | 0.416    | 0.319   | 0.0001   | 0.0003  | ↑ Coverage, ↓ D95 (very low dose)  |
| 88     | 1.000    | 1.000   | 1.000    | 1.000   | Same coverage and D95              |
| 89     | 1.000    | 1.000   | 1.000    | 1.000   | Same coverage and D95              |
| 90     | 0.794    | 0.846   | 0.064    | 0.183   | ↓ Coverage, ↓ D95                  |

Values are rounded to three decimal places for readability.

---

## 🔍 Pattern Over These 20 Cases

- **Coverage advantage:**  
  - DQN has **higher PTV coverage in 16/20** cases.  
  - **2/20** are ties (patients 88, 89, both with perfect target coverage).  
  - **2/20** cases (49, 90) show higher coverage for the equiangular baseline.

- **D95 advantage:**  
  - DQN has **higher D95 in 11/20** cases.  
  - **3/20** are essentially equal (patients 4, 88, 89).  
  - **6/20** cases (28, 32, 49, 50, 84, 90) have higher D95 for the equiangular plan, though several of these operate in a very low-dose regime where absolute D95 values are small.

- **Strong cases:**  
  - Patients **74, 80, 81, 83, 88, 89** illustrate near-complete or complete coverage with high D95, where DQN clearly dominates or matches the baselines.
  - Patient **74** is a canonical example: **coverage ≈ 0.99, D95 ≈ 0.90** vs equiangular **coverage ≈ 0.89, D95 ≈ 0.31**.

- **Failure / edge cases:**  
  - Patients **49** and **90** are genuine counterexamples where the equiangular baseline achieves noticeably higher coverage and D95.
  - Patients **28, 32, 84** show the model pushing coverage up but with very low absolute D95 values, reflecting challenging geometry or limitations of the surrogate dose model.

---

## 🩻 Link to Figures

For these 20 patients, corresponding qualitative figures (CT + dose overlays and DVH curves) are stored in:

- `figures/success_cases/`
- `figures/typical_cases/`
- `figures/failure_cases/`
- `figures/anomaly_cases/`

These visualizations complement the quantitative metrics above, showing how the DQN generally keeps high-dose regions within the PTV while reducing unnecessary exposure to nearby OARs, with a few clearly documented failure modes.

---

## 📁 Output Files

- `results/test_results.csv` — metrics for all 100 patients  
- `/figures/` — dose maps and DVH plots  
- Model weights: `best_dqn_model.pt`  
- Source code: `train.py`, `eval_main.py`  

---

## 📝 Notes

- All dose maps normalized to [0, 1]
- Evaluation performed slice-wise (single axial slice per patient)
- DVH curves display dose distributions across PTV and OARs
