import os, numpy as np, torch, matplotlib.pyplot as plt, pandas as pd
from train import policy, load_patients, organs, shape, max_beams, generate_beam, beam_angles
from baselines import equiangular_beams, random_beam_sets, heuristic_beams_from_ptv, greedy_beams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Using device:", device)

test_dir = r"C:\BAO\openkbp_patient_data\test-pats"
print("📂 Test data:", test_dir)

from torch.serialization import add_safe_globals
add_safe_globals([np.core.multiarray.scalar])

checkpoint = torch.load("best_dqn_model.pt", map_location=device, weights_only=False)
policy.load_state_dict(checkpoint["policy_state_dict"])
policy.eval()
print("✅ Best model loaded")

def evaluate_beam_set_on_patient(pt, beam_indices, beam_angles, max_beams=5):
    dose = np.zeros((pt.shape[1], pt.shape[2]), dtype=np.float32)
    for a in beam_indices:
        beam = generate_beam(beam_angles[a], pt[1])
        dose += beam
        dose = np.clip(dose, 0, 1)
    dose = dose / (dose.max() + 1e-8)
    return dose


def compute_d95(dose, mask):
    vals = dose[mask > 0]
    return np.percentile(vals,5) if len(vals)>0 else 0

def compute_metrics(dose, pt, organs):
    ptv = pt[1]
    oars = [pt[i] for i in range(2, pt.shape[0])]
    d95 = compute_d95(dose, ptv)
    coverage = (dose[ptv > 0] > 0.5).mean() if ptv.sum() > 0 else 0
    oar_means = [(dose[o > 0].mean() if o.sum() > 0 else 0) for o in oars]

    metrics = {
        "D95": float(d95),
        "Coverage": float(coverage),
        "SpinalCord": float(oar_means[0]),
        "Brainstem": float(oar_means[1]),
        "LtParotid": float(oar_means[2]),
        "RtParotid": float(oar_means[3]),
        "Mandible": float(oar_means[4]),
    }
    return metrics

def build_state(patient_data, dose_map):
    return np.vstack([patient_data, dose_map[np.newaxis,:,:]]).astype(np.float32)

print("📦 Loading test patients...")
test_patients = load_patients(test_dir, organs, shape)
print(f"🧪 Test patients loaded: {len(test_patients)}")

results = []

for idx, pt in enumerate(test_patients):
    ptv = pt[1]
    oars = [pt[i] for i in range(2, pt.shape[0])]

    # ---------- DQN POLICY ----------
    dose = np.zeros(shape[1:], dtype=np.float32)
    state = build_state(pt, dose)
    selected = []

    for _ in range(max_beams):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        q = policy(s).detach().cpu().numpy()[0]
        for a in selected:
            q[a] = -1e9
        a = int(np.argmax(q))
        selected.append(a)

        beam = generate_beam(beam_angles[a], ptv)
        dose += beam
        dose = np.clip(dose, 0, 1)
        state = build_state(pt, dose)

    dose = dose / dose.max() if dose.max() > 0 else dose
    metrics_dqn = compute_metrics(dose, pt, organs)
    metrics_dqn.update({
        "patient": idx,
        "method": "DQN",
        "Beams": [int(beam_angles[a]) for a in selected]
    })
    results.append(metrics_dqn)

    print(f"\n🧾 Patient {idx+1}/{len(test_patients)}")
    print("DQN beams →", [beam_angles[a] for a in selected])
    print(f"DQN D95 = {metrics_dqn['D95']:.4f}, Coverage = {metrics_dqn['Coverage']*100:.2f}%")

    # ---------- EQUIANGULAR BASELINE ----------
    eq_idx = equiangular_beams(max_beams, beam_angles)
    dose_eq = evaluate_beam_set_on_patient(pt, eq_idx, beam_angles, max_beams)
    metrics_eq = compute_metrics(dose_eq, pt, organs)
    metrics_eq.update({
        "patient": idx,
        "method": "Equiangular",
        "Beams": [int(beam_angles[a]) for a in eq_idx]
    })
    results.append(metrics_eq)

    # ---------- HEURISTIC BASELINE ----------
    h_idx = heuristic_beams_from_ptv(ptv, max_beams)
    dose_h = evaluate_beam_set_on_patient(pt, h_idx, beam_angles, max_beams)
    metrics_h = compute_metrics(dose_h, pt, organs)
    metrics_h.update({
        "patient": idx,
        "method": "Heuristic",
        "Beams": [int(beam_angles[a]) for a in h_idx]
    })
    results.append(metrics_h)

    # ---------- RANDOM BASELINE (mean of several sets) ----------
    n_random = 20  # you can increase later
    rand_sets = random_beam_sets(max_beams, beam_angles, n_samples=n_random, seed=idx)
    covs, d95s = [], []
    for s_beams in rand_sets:
        dose_r = evaluate_beam_set_on_patient(pt, s_beams, beam_angles, max_beams)
        m_r = compute_metrics(dose_r, pt, organs)
        covs.append(m_r["Coverage"])
        d95s.append(m_r["D95"])
    results.append({
        "patient": idx,
        "method": "RandomMean",
        "D95": float(np.mean(d95s)),
        "Coverage": float(np.mean(covs)),
        "SpinalCord": None,
        "Brainstem": None,
        "LtParotid": None,
        "RtParotid": None,
        "Mandible": None,
        "Beams": None
    })

    # ---------- OPTIONAL: plotting (only for a few patients) ----------
    if idx < 3:  # plot only first 3 patients to avoid spam
        ct = pt[0]
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)

        plt.figure(figsize=(6,6))
        plt.imshow(ct, cmap="gray", alpha=1.0)
        img = plt.imshow(dose, cmap="hot", alpha=0.55)

        ptv_edge = np.logical_xor(ptv, np.logical_and(
            np.pad(ptv[1:,:], ((0,1),(0,0))),
            np.pad(ptv[:,1:], ((0,0),(0,1)))
        ))
        plt.contour(ptv_edge, colors="cyan", linewidths=1.5)
        plt.title(f"Patient {idx} — DQN Dose Map\n(Tumor = Cyan Boundary)")
        plt.colorbar(img, label="Normalized Dose")
        plt.axis("off")
        plt.show()

        # DVH
        plt.figure(figsize=(6,5))
        ptv_vals = dose[ptv > 0].flatten()
        plt.hist(ptv_vals, bins=80, density=True, histtype="step",
                 linewidth=2, label="PTV (Target)")

        for oar, name in zip(oars, organs):
            if oar.sum() > 0:
                vals = dose[oar > 0].flatten()
                plt.hist(vals, bins=80, density=True, histtype="step",
                         linewidth=1.3, label=name)

        plt.xlabel("Dose (normalized)")
        plt.ylabel("Volume Fraction")
        plt.title(f"DVH — Patient {idx} (DQN)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.show()

df = pd.DataFrame(results)
df.to_csv("test_results.csv",index=False)
print("\n✅ Results saved: test_results.csv")
print("\n📊 Overall Mean:")
print(df.mean(numeric_only=True))
print("\n📊 Mean by method:")
print(df.groupby("method")[["Coverage", "D95"]].mean())
