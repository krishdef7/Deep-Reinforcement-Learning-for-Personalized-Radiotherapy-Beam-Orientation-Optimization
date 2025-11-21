import os, numpy as np, torch, matplotlib.pyplot as plt, pandas as pd

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

def compute_d95(dose, mask):
    vals = dose[mask > 0]
    return np.percentile(vals,5) if len(vals)>0 else 0

def build_state(patient_data, dose_map):
    return np.vstack([patient_data, dose_map[np.newaxis,:,:]]).astype(np.float32)

print("📦 Loading test patients...")
test_patients = load_patients(test_dir, organs, shape)
print(f"🧪 Test patients loaded: {len(test_patients)}")

results = []

for idx, pt in enumerate(test_patients):
    ptv = pt[1]
    oars = [pt[i] for i in range(2, pt.shape[0])]
    dose = np.zeros((128,128),dtype=np.float32)
    state = build_state(pt, dose)
    selected = []

    for _ in range(max_beams):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        q = policy(s).detach().cpu().numpy()[0]
        for a in selected: q[a] = -1e9
        a = int(np.argmax(q))
        selected.append(a)

        beam = generate_beam(beam_angles[a], ptv)
        dose += beam
        dose = np.clip(dose,0,1)
        state = build_state(pt,dose)

    dose = dose / dose.max() if dose.max()>0 else dose

    d95 = compute_d95(dose, ptv)
    cov = (dose[ptv>0] > 0.5).mean() if ptv.sum()>0 else 0
    oar_mean = [(dose[o>0].mean() if o.sum()>0 else 0) for o in oars]

    results.append({
        "patient": idx,
        "D95": d95,
        "Coverage": cov,
        "SpinalCord": oar_mean[0],
        "Brainstem": oar_mean[1],
        "LtParotid": oar_mean[2],
        "RtParotid": oar_mean[3],
        "Mandible": oar_mean[4],
        "Beams": [beam_angles[a] for a in selected]
    })

    print(f"\n🧾 Patient {idx+1}/{len(test_patients)}")
    print("Beams →", [beam_angles[a] for a in selected])
    print(f"D95 = {d95:.4f}, Coverage = {cov*100:.2f}%")

    # ---- Clinical Dose Map ----
    ct = pt[0]
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)

    plt.figure(figsize=(6,6))
    plt.imshow(ct,cmap="gray",alpha=1.0)
    img = plt.imshow(dose,cmap="hot",alpha=0.55)

    # PTV edge
    ptv_edge = np.logical_xor(ptv, np.logical_and(
        np.pad(ptv[1:,:],((0,1),(0,0))),
        np.pad(ptv[:,1:],((0,0),(0,1)))
    ))
    plt.contour(ptv_edge,colors="cyan",linewidths=1.5)

    plt.title(f"Patient {idx} — Clinical Dose Map\n(Tumor = Cyan Boundary)")
    plt.colorbar(img,label="Normalized Dose")
    plt.axis("off")
    plt.show()

    # ---- Clinical DVH ----
    plt.figure(figsize=(6,5))
    ptv_vals = dose[ptv>0].flatten()
    plt.hist(ptv_vals,bins=80,density=True,histtype="step",linewidth=2,label="PTV (Target)")

    for oar,name in zip(oars,organs):
        if oar.sum()>0:
            vals = dose[oar>0].flatten()
            plt.hist(vals,bins=80,density=True,histtype="step",linewidth=1.3,label=name)

    plt.xlabel("Dose (normalized)")
    plt.ylabel("Volume Fraction")
    plt.title(f"DVH — Patient {idx}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.show()

df = pd.DataFrame(results)
df.to_csv("test_results.csv",index=False)
print("\n✅ Results saved: test_results.csv")
print("\n📊 Overall Mean:")
print(df.mean(numeric_only=True))
