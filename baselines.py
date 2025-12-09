# baselines.py
import numpy as np
from collections import defaultdict

# beam_angles: np.arange(0,360,10) assumed externally
# generate_beam(ptv, angle) and get_heuristic_angles(ptv, n_beams) should exist in your code.
# If they live in train.py or eval_main.py, import them like:
# from train import generate_beam, get_heuristic_angles
# or change import as needed.

def equiangular_beams(n_beams=5, beam_angles=np.arange(0,360,10)):
    angles = np.linspace(0, 360, n_beams, endpoint=False)
    indices = [int(a) // 10 for a in angles]
    return indices

def random_beam_sets(n_beams=5, beam_angles=np.arange(0,360,10), n_samples=100, seed=0):
    rng = np.random.RandomState(seed)
    L = len(beam_angles)
    results = []
    for _ in range(n_samples):
        res = rng.choice(L, size=n_beams, replace=False).tolist()
        results.append(res)
    return results

def heuristic_beams_from_ptv(ptv_mask, n_beams=5):
    """
    Wrapper that calls get_heuristic_angles if available.
    Returns indices corresponding to beam_angles step 10 deg.
    """
    try:
        # try to use user-defined heuristic function
        from eval_main import get_heuristic_angles  # adapt if function is elsewhere
    except Exception:
        try:
            from train import get_heuristic_angles
        except Exception:
            # fallback: equiangular
            return equiangular_beams(n_beams)
    angles_idx = get_heuristic_angles(ptv_mask, n_beams)
    # ensure they are ints and valid indices
    return [int(a) % len(np.arange(0,360,10)) for a in angles_idx]

def greedy_beams(pt, n_beams=5, beam_angles=np.arange(0,360,10), score_fn=None):
    """
    Greedy: choose each beam to maximize immediate coverage gain (or custom score_fn).
    pt: patient array (channels, H, W), assumes pt[1] is PTV mask
    """
    try:
        from eval_main import generate_beam
    except Exception:
        try:
            from train import generate_beam
        except Exception:
            raise RuntimeError("generate_beam not found in train.py or eval_main.py. Import it or adjust path.")

    dose = np.zeros((pt.shape[1], pt.shape[2]), dtype=np.float32)
    selected = []
    for _ in range(n_beams):
        best_a, best_score = None, -1e9
        for i in range(len(beam_angles)):
            if i in selected: 
                continue
            beam = generate_beam(beam_angles[i], pt[1])
            new_dose = np.clip(dose + beam, 0, 1)
            if score_fn is None:
                prev_cov = (dose[pt[1] > 0] > 0.5).mean() if pt[1].sum()>0 else 0
                new_cov = (new_dose[pt[1] > 0] > 0.5).mean() if pt[1].sum()>0 else 0
                sc = new_cov - prev_cov
            else:
                sc = score_fn(dose, new_dose, pt[1])
            if sc > best_score:
                best_score, best_a = sc, i
        if best_a is None:
            # fallback if something goes wrong
            remaining = [i for i in range(len(beam_angles)) if i not in selected]
            if not remaining:
                break
            best_a = remaining[0]
        selected.append(best_a)
        dose = np.clip(dose + generate_beam(beam_angles[best_a], pt[1]), 0, 1)
    return selected
