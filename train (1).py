import os, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from scipy.ndimage import gaussian_filter
from skimage.draw import line_nd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuration
base_path = r"C:\BAO\openkbp_patient_data"
train_dir = os.path.join(base_path, "train-pats")
val_dir = os.path.join(base_path, "validation-pats")

organs = ["SpinalCord", "Brainstem", "LeftParotid", "RightParotid", "Mandible"]
beam_angles = np.arange(0, 360, 10)
max_beams = 5
epochs = 50
batch_size = 32
gamma = 0.95
epsilon, eps_decay, eps_min = 0.9, 0.98, 0.1
lr = 1e-4
memory_size = 3000
target_update = 5
shape = (128, 128, 128)
warm_start_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading utilities
def load_voxel_csv(path, shape=(128,128,128)):
    if not os.path.exists(path):
        return np.zeros(shape, dtype=np.float32)
    try:
        df = pd.read_csv(path, header=None, skiprows=1)
        df = df.dropna(how='all')
        if df.shape[1] == 1 or (df.shape[1] > 1 and df.iloc[:,1].isna().all()):
            idx = df.iloc[:,0].astype(int).clip(0, np.prod(shape)-1).values
            arr = np.zeros(np.prod(shape), dtype=np.float32)
            arr[idx] = 1.0
            return arr.reshape(shape)
        idx = df.iloc[:,0].astype(int).clip(0, np.prod(shape)-1).values
        vals = df.iloc[:,1].astype(float).values
        arr = np.zeros(np.prod(shape), dtype=np.float32)
        arr[idx] = vals
        return arr.reshape(shape)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.zeros(shape, dtype=np.float32)


def load_patients(base_dir, organs, shape=(128,128,128)):
    patients = []
    for name in sorted(os.listdir(base_dir)):
        p_dir = os.path.join(base_dir, name)
        if not os.path.isdir(p_dir):
            continue
        ct = load_voxel_csv(os.path.join(p_dir, "ct.csv"), shape)
        if np.max(ct) > 0:
            ct = (ct - np.min(ct)) / (np.max(ct) - np.min(ct) + 1e-8)
        
        # Try to load PTV
        ptv = None
        for p in ["PTV70.csv", "PTV63.csv", "PTV56.csv"]:
            p_path = os.path.join(p_dir, p)
            if os.path.exists(p_path):
                ptv = load_voxel_csv(p_path, shape)
                break
        if ptv is None or ptv.sum() == 0:
            continue
        
        # Get middle slice
        z_slices = np.where(ptv.sum(axis=(1,2)) > 0)[0]
        if len(z_slices) == 0:
            continue
        mid = int(np.median(z_slices))
        ct_slice = ct[mid,:,:]
        ptv_slice = ptv[mid,:,:]
        
        # Load OAR slices
        oar_slices = [load_voxel_csv(os.path.join(p_dir, f"{o}.csv"), shape)[mid,:,:] for o in organs]
        patient = np.stack([ct_slice, ptv_slice] + oar_slices, axis=0).astype(np.float32)
        patients.append(patient)
    print(f"Loaded {len(patients)} valid patient slices from {base_dir}")
    return patients


print("\nLoading patient data...")
train_patients = load_patients(train_dir, organs, shape)
val_patients = load_patients(val_dir, organs, shape)
print(f"Training: {len(train_patients)} | Validation: {len(val_patients)}")

# DQN model
class DQN_CNN(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Beam generation
def generate_beam(angle, ptv_mask=None, shape=(128,128)):
    h, w = shape
    beam = np.zeros((h, w), dtype=np.float32)
    if ptv_mask is not None and ptv_mask.sum() > 0:
        y, x = np.where(ptv_mask > 0)
        cy, cx = int(y.mean()), int(x.mean())
    else:
        cy, cx = h//2, w//2
    L = max(h, w)
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), np.sin(rad)
    s = np.clip((np.array([cy, cx]) - L*np.array([dy, dx])).astype(int), 0, [h-1,w-1])
    e = np.clip((np.array([cy, cx]) + L*np.array([dy, dx])).astype(int), 0, [h-1,w-1])
    rr, cc = line_nd(s, e)
    beam[rr, cc] = 1.0
    beam = gaussian_filter(beam, sigma=2.0)
    return beam / (beam.max() + 1e-8)

def compute_d95(dose, mask):
    """D95: dose received by 95% of target volume"""
    vals = dose[mask > 0]
    return np.percentile(vals, 5) if len(vals) > 0 else 0.0


def compute_step_reward(prev_dose, new_dose, ptv, oars, is_terminal=False):
    # Normalize doses
    prev_d = prev_dose / (prev_dose.max() + 1e-8) if prev_dose.max() > 0 else prev_dose.copy()
    new_d = new_dose / (new_dose.max() + 1e-8) if new_dose.max() > 0 else new_dose.copy()
    
    tumor_pixels = np.sum(ptv)
    if tumor_pixels == 0:
        return -10.0
    
    # Coverage improvement reward
    prev_coverage = np.sum((prev_d * ptv) > 0.5) / tumor_pixels
    new_coverage  = np.sum((new_d  * ptv) > 0.5) / tumor_pixels
    coverage_gain = new_coverage - prev_coverage
    step_reward = 200 * coverage_gain
    
    # OAR penalty
    oar_weights = [2.5, 2.5, 0.6, 0.6, 1.2]
    for w, oar in zip(oar_weights, oars):
        if oar.sum() > 0:
            prev_oar = np.sum(prev_d * oar) / oar.sum()
            new_oar  = np.sum(new_d  * oar) / oar.sum()
            oar_increase = new_oar - prev_oar
            step_reward -= w * 15 * oar_increase
    
    # Terminal bonuses
    if is_terminal:
        final_coverage = new_coverage
        d95 = compute_d95(new_d, ptv)
        mean_tumor = np.sum(new_d * ptv) / tumor_pixels
        
        # Coverage bonuses
        if final_coverage > 0.95: step_reward += 300
        elif final_coverage > 0.90: step_reward += 200
        elif final_coverage > 0.85: step_reward += 150
        elif final_coverage > 0.80: step_reward += 100
        elif final_coverage > 0.70: step_reward += 50
        else: step_reward -= 50
        
        # D95 bonus
        step_reward += 200 * d95
        step_reward += 30 * mean_tumor
        
        # Max OAR penalty
        max_oar_dose = max([new_d[oar > 0].max() if oar.sum() > 0 else 0 for oar in oars])
        if max_oar_dose > 0.7:
            step_reward -= 100 * (max_oar_dose - 0.7)
    
    return float(np.clip(step_reward, -100.0, 500.0))

def build_state(patient_data, dose_map):
    return np.vstack([patient_data, dose_map[np.newaxis, :, :]]).astype(np.float32)

def get_heuristic_angles(ptv_mask, n_beams=5):
    if ptv_mask.sum() == 0:
        angles_deg = np.linspace(0, 360, n_beams, endpoint=False)
        return [int(a)//10 for a in angles_deg]
    y, x = np.where(ptv_mask > 0)
    cy, cx = y.mean(), x.mean()
    x_std, y_std = x.std(), y.std()
    if x_std > 1.5 * y_std:
        base_angles = [0, 45, 90, 270, 315]
    elif y_std > 1.5 * x_std:
        base_angles = [0, 72, 144, 216, 288]
    else:
        base_angles = [0, 72, 144, 216, 288]
    return [angle // 10 for angle in base_angles[:n_beams]]

def select_action(policy, state, selected_angles, epsilon, beam_angles, device):
    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    if random.random() < epsilon:
        available = [i for i in range(len(beam_angles)) if i not in selected_angles]
        if len(available) == 0:
            return random.randint(0, len(beam_angles) - 1)
        return random.choice(available)
    else:
        with torch.no_grad():
            q_values = policy(s_t).cpu().numpy()[0]
            q_values_masked = q_values.copy()
            for idx in selected_angles:
                q_values_masked[idx] = -1e9
            return int(np.argmax(q_values_masked))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Initialize model
in_channels = 8
action_dim = len(beam_angles)
policy = DQN_CNN(in_channels, action_dim).to(device)
target = DQN_CNN(in_channels, action_dim).to(device)
target.load_state_dict(policy.state_dict())
optimizer = optim.Adam(policy.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
memory = ReplayBuffer(memory_size)

best_val_d95, best_val_coverage = -np.inf, -np.inf

print(f"\nModel Architecture:")
print(f"  Input: {in_channels} channels x 128x128")
print(f"  Output: {action_dim} beam angles")
print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
print("\nStarting training...\n")

training_history = {'loss': [], 'reward': [], 'val_d95': [], 'val_coverage': [], 'epsilon': [], 'lr': []}

# Training loop
for ep in range(epochs):
    random.shuffle(train_patients)
    total_loss, ep_rewards, ep_coverages, ep_d95s = 0, [], [], []
    policy.train()
    use_warm_start = (ep < warm_start_epochs)

    with tqdm(total=len(train_patients), desc=f"Epoch {ep+1}/{epochs}", 
              ncols=140, mininterval=1.0) as pbar:
        
        for pt_idx, pt in enumerate(train_patients):
            ptv = pt[1]
            oars = [pt[i] for i in range(2, pt.shape[0])]
            dose = np.zeros((128, 128), dtype=np.float32)
            state = build_state(pt, dose)
            
            if use_warm_start:
                heuristic_angles = get_heuristic_angles(ptv, max_beams)
            
            selected_angles = []
            for step in range(max_beams):
                prev_dose = dose.copy()
                
                if use_warm_start and random.random() < 0.7:
                    a = heuristic_angles[step]
                    if a in selected_angles:
                        a = select_action(policy, state, selected_angles, 0.3, beam_angles, device)
                else:
                    a = select_action(policy, state, selected_angles, epsilon, beam_angles, device)
                
                selected_angles.append(a)
                beam = generate_beam(beam_angles[a], ptv)
                dose += beam
                dose = np.clip(dose, 0, 1)
                
                is_terminal = (step == max_beams - 1)
                r = compute_step_reward(prev_dose, dose, ptv, oars, is_terminal)
                ep_rewards.append(r)
                
                next_state = build_state(pt, dose)
                memory.push(state, a, r, next_state, is_terminal)
                state = next_state
                
                # DQN update
                if len(memory) >= batch_size:
                    batch = memory.sample(batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.FloatTensor(np.array(states)).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(device)
                    dones = torch.FloatTensor(dones).to(device)
                    
                    q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        next_actions = policy(next_states).argmax(dim=1)
                        next_q_values = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        target_q = rewards + gamma * next_q_values * (1 - dones)
                    
                    loss = nn.SmoothL1Loss()(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
            
            # Compute metrics
            dose_norm = dose / dose.max() if dose.max() > 0 else dose
            coverage = (dose_norm[ptv > 0] > 0.5).mean() if ptv.sum() > 0 else 0
            d95 = compute_d95(dose_norm, ptv)
            ep_coverages.append(coverage)
            ep_d95s.append(d95)
            
            if pt_idx % 5 == 0:
                avg_r = np.mean(ep_rewards[-50:]) if len(ep_rewards) > 0 else 0.0
                avg_cov = np.mean(ep_coverages[-10:]) if len(ep_coverages) > 0 else 0.0
                avg_d95 = np.mean(ep_d95s[-10:]) if len(ep_d95s) > 0 else 0.0
                pbar.set_postfix(
                    Loss=f"{total_loss/(pbar.n+1):.2f}",
                    Reward=f"{avg_r:.1f}",
                    Cov=f"{avg_cov:.1%}",
                    D95=f"{avg_d95:.3f}",
                    Eps=f"{epsilon:.3f}",
                    WS="✓" if use_warm_start else ""
                )
            pbar.update(1)
    
    epsilon = max(eps_min, epsilon * eps_decay)
    
    if (ep + 1) % target_update == 0:
        target.load_state_dict(policy.state_dict())
        print(f"Target network updated at epoch {ep+1}")

    # Validation
    policy.eval()
    val_d95s, val_coverages, val_rewards = [], [], []
    with torch.no_grad():
        for v in val_patients:
            ptv = v[1]
            if ptv.sum() == 0:
                continue
            oars = [v[i] for i in range(2, v.shape[0])]
            dose = np.zeros((128, 128), dtype=np.float32)
            state = build_state(v, dose)
            selected_angles = []
            
            for step in range(max_beams):
                s_v = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_vals = policy(s_v).cpu().numpy()[0]
                q_vals_masked = q_vals.copy()
                for idx in selected_angles:
                    q_vals_masked[idx] = -1e9
                a = int(np.argmax(q_vals_masked))
                selected_angles.append(a)
                prev_dose = dose.copy()
                beam = generate_beam(beam_angles[a], ptv)
                dose += beam
                dose = np.clip(dose, 0, 1)
                state = build_state(v, dose)
                is_terminal = (step == max_beams - 1)
                r = compute_step_reward(prev_dose, dose, ptv, oars, is_terminal)
                val_rewards.append(r)
            
            dose = dose / dose.max() if dose.max() > 0 else dose
            val_d95s.append(compute_d95(dose, ptv))
            val_coverages.append((dose[ptv > 0] > 0.5).mean() if ptv.sum() > 0 else 0)
    
    val_d95 = np.mean(val_d95s) if len(val_d95s) > 0 else 0.0
    val_coverage = np.mean(val_coverages) if len(val_coverages) > 0 else 0.0
    val_reward = np.mean(val_rewards) if len(val_rewards) > 0 else 0.0
    
    training_history['loss'].append(total_loss / len(train_patients))
    training_history['reward'].append(np.mean(ep_rewards))
    training_history['val_d95'].append(val_d95)
    training_history['val_coverage'].append(val_coverage)
    training_history['epsilon'].append(epsilon)
    training_history['lr'].append(optimizer.param_groups[0]['lr'])
    
    scheduler.step(val_d95)

    # Save best model
    if val_d95 > best_val_d95:
        best_val_d95 = val_d95
        best_val_coverage = val_coverage
        torch.save({
            'epoch': ep,
            'policy_state_dict': policy.state_dict(),
            'val_d95': val_d95,
            'val_coverage': val_coverage
        }, 'best_dqn_model.pt')
        print(f"New best model saved! D95={val_d95:.4f}")

    # Print epoch summary
    print(f"\nEpoch {ep+1} Summary:")
    print(f"  Train: Reward={np.mean(ep_rewards):.2f} | Coverage={np.mean(ep_coverages)*100:.2f}% "
          f"| D95={np.mean(ep_d95s):.4f}")
    print(f"  Val:   Reward={val_reward:.2f} | Coverage={val_coverage*100:.2f}% "
          f"| D95={val_d95:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Epsilon: {epsilon:.4f}\n")

print("\nTraining completed!")
print(f"Best Validation D95: {best_val_d95:.4f}")
print(f"Best Validation Coverage: {best_val_coverage*100:.2f}%")