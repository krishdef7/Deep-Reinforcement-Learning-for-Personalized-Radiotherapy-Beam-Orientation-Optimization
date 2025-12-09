# 📈 Evaluation Summary

### Dataset
- 100 held-out test cases (OpenKBP head-and-neck)
- Inputs: CT + PTV + 5 OAR structures
- Metrics:
  - **D95** (coverage of prescription dose)
  - **PTV Coverage** (fraction receiving ≥threshold dose)
  - **Mean OAR dose**

---

## 🧪 Mean Performance (100 Patients)

| Method       | Coverage | D95   |
|--------------|----------|-------|
| **DQN (ours)** | **0.8058** | **0.2405** |
| Equiangular  | 0.6867   | 0.1207 |
| Heuristic    | 0.6397   | 0.0949 |
| Random       | 0.5883   | 0.0554 |

**Observation:**  
The DQN systematically improves PTV coverage and D95 relative to equiangular and heuristic baselines.

---

## 🩻 Dose Distribution Examples

Representative samples (see `/figures_sample/`):

| Patient | Coverage | D95 |
|--------|----------|-----|
| #8     | 1.0000   | 1.0000 |
| #14    | 0.9030   | 0.3203 |
| #56    | 0.8910   | 0.3596 |
| #81    | 0.9780   | 0.6691 |
| #90    | 1.0000   | 1.0000 |

---

## 📁 Output Files

- `results/test_results.csv` — metrics for all 100 patients  
- `/figures_sample/` — dose maps and DVH plots  
- Model weights: `best_dqn_model.pt`  
- Source code: `train.py`, `eval_main.py`  

---

## 📝 Notes

- All dose maps normalized to [0, 1]
- Evaluation performed slice-wise (single axial slice per patient)
- DVH curves display dose distributions across PTV and OARs
