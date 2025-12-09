# Summary of Evaluation Results

### Dataset
- 100 test CT cases (OpenKBP)
- Target (PTV) + 5 OARs

### Metrics
- D95 (Target Coverage)
- Coverage (% of target above threshold)
- Mean Dose to OARs

---

## 📊 Mean Performance (100 Patients)

| Method       | Coverage | D95   |
|--------------|----------|-------|
| **DQN**      | **0.8058** | **0.2405** |
| Equiangular  | 0.6867   | 0.1207 |
| Heuristic    | 0.6397   | 0.0949 |
| Random       | 0.5883   | 0.0554 |

**Conclusion:**  
DQN demonstrates consistent improvements in dose coverage compared to conventional baselines.

---

## 🩻 Dose Distribution Examples

Representative cases (see `/figures_sample/`):

| Patient | Coverage | D95 |
|--------|----------|-----|
| #8     | 100.0%   | 1.0000 |
| #14    | 90.3%    | 0.3203 |
| #56    | 89.1%    | 0.3596 |
| #81    | 97.8%    | 0.6691 |
| #90    | 100.0%   | 1.0000 |

---

## 🛠 Files
- `test_results.csv` — raw results for all patients
- `/figures_sample/` — qualitative dose + DVH visualization
- Model weights: `best_dqn_model.pt`
- Code: `train.py`, `eval_main.py`

---

## 📝 Notes
- All data normalized (0–1)
- Dose shown over a single CT slice
- OAR doses shown in DVH curves
