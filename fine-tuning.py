# fine_tuning.py
import re
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from braindecode.models import LUNA
from braindecode.util import set_random_seeds
from tqdm import tqdm
import random
import csv
import yaml
import wandb

# 1. Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract parameters
BIDS_ROOTS = [Path(p) for p in config["bids_roots"]]
CKPT_PATH = Path(config["ckpt_path"])
SAVE_DIR = Path(config["save_dir"])
TEST_SUBJECTS = set(config["test_subjects"])
TARGET_SFREQ = config["target_sfreq"]
WINDOW_SEC = config["window_sec"]
N_TIMES = int(TARGET_SFREQ * WINDOW_SEC)
N_CHANS_OUT = config["n_chans_out"]
OVERLAP = config["overlap"]
N_EPOCHS = config["n_epochs"]
FROZEN_EPOCHS = config["frozen_epochs"]
LR_HEAD = config["lr_head"]
LR_BACKBONE = config["lr_backbone"]
BATCH_SIZE = config["batch_size"]
SEED = config["seed"]
EARLY_STOPPING_PATIENCE = config["early_stopping_patience"]
GRAD_ACCUM_STEPS = config.get("grad_accum_steps", 1)  # ← NUOVO: gradient accumulation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seeds(SEED, cuda=torch.cuda.is_available())
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 2. Initialize wandb run
wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=config["wandb"].get("run_name"),
    tags=config["wandb"].get("tags", []),
    config=config
)
wandb.config.update({
    "device": DEVICE,
    "n_times": N_TIMES,
    "n_chans_out": N_CHANS_OUT,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "window_sec": WINDOW_SEC
})

# 3. Dataset discovery and split
print("DATASET DISCOVERY")
ALL_SUBJECTS = {}
for root in BIDS_ROOTS:
    if not root.exists():
        print(f"{root} NOT FOUND, skipping")
        continue
    subjects = sorted([d.name for d in root.glob("sub-*") if d.is_dir()])
    ALL_SUBJECTS[root.name] = subjects
    print(f"{root.name}: {len(subjects)} subjects")

total_subjects = sum(len(subs) for subs in ALL_SUBJECTS.values())
print(f"\nTotal subjects: {total_subjects}\n")
print(f"Test subjects (held-out): {', '.join(TEST_SUBJECTS)}\n")

VAL_SUBJECTS = []
TRAIN_SUBJECTS = []

for ds_name, subs in ALL_SUBJECTS.items():
    eligible = [s for s in subs if s not in TEST_SUBJECTS]
    if len(eligible) == 0:
        continue
    rng_shuffle = random.Random(SEED)
    eligible_shuffled = rng_shuffle.sample(eligible, len(eligible))
    n_val = max(1, int(len(eligible_shuffled) * 0.2))
    val_this = eligible_shuffled[:n_val]
    train_this = eligible_shuffled[n_val:]
    VAL_SUBJECTS.extend(val_this)
    TRAIN_SUBJECTS.extend(train_this)
    print(f"{ds_name}: eligible={len(eligible)}, val={len(val_this)}, train={len(train_this)}")

def get_subject_paths(subject_list):
    paths = []
    for root in BIDS_ROOTS:
        if not root.exists():
            continue
        for subj in subject_list:
            subj_path = root / subj
            if subj_path.exists():
                paths.append(subj_path)
    return paths

train_paths = get_subject_paths(TRAIN_SUBJECTS)
val_paths   = get_subject_paths(VAL_SUBJECTS)

print(f"\nTraining:   {len(train_paths)} subjects")
print(f"Validation: {len(val_paths)} subjects")
print(f"Test:       {len(TEST_SUBJECTS)} subjects (hold-out)\n")

# 4. Parsing events
ONSET_RE = re.compile(
    r"sz event|seizure.*onset|ictal onset|\bonset\b|\bsz\b|"
    r"poss.*onset|clinical onset|electrographic onset",
    re.I
)
OFFSET_RE = re.compile(
    r"electrographic end|offset|sz end|seizure off|devolution|"
    r"ending|clinical end|seizure end",
    re.I
)

def parse_seizure_intervals(events_tsv: Path):
    if not events_tsv.exists():
        return []
    df = pd.read_csv(events_tsv, sep="\t")
    onsets, offsets = [], []
    for _, row in df.iterrows():
        label = str(row.get("trial_type", ""))
        t = float(row["onset"])
        if ONSET_RE.search(label):
            onsets.append(t)
        elif OFFSET_RE.search(label):
            offsets.append(t)
    intervals = []
    for on in onsets:
        later = [o for o in offsets if o > on]
        off = min(later) if later else on + 60.0
        intervals.append((on, off))
    return intervals

# 5. Loading raw
def load_run(file_path: Path):
    if file_path.suffix == ".edf":
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
    elif file_path.suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(file_path), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    ch_file = file_path.parent / file_path.name.replace("_ieeg.edf", "_channels.tsv").replace("_ieeg.vhdr", "_channels.tsv")
    if ch_file.exists():
        ch_df = pd.read_csv(ch_file, sep="\t")
        if "status" in ch_df.columns:
            good = ch_df[ch_df["status"] == "good"]["name"].tolist()
            available = [c for c in good if c in raw.ch_names]
            if available:
                raw.pick(available)
    
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1:
        raw.resample(TARGET_SFREQ, verbose=False)
    return raw

# 6. Windowing with deterministic channel selection
def extract_windows(data: np.ndarray, sfreq: int, ictal_intervals: list, file_seed: int):
    n_chans, total = data.shape
    step = int(N_TIMES * (1 - OVERLAP))
    rng_file = np.random.default_rng(file_seed)
    windows = []
    
    for start in range(0, total - N_TIMES, step):
        end = start + N_TIMES
        t_start, t_end = start / sfreq, end / sfreq
        
        label = 0
        for (on, off) in ictal_intervals:
            overlap_s = max(0, min(t_end, off) - max(t_start, on))
            if overlap_s / (t_end - t_start) > 0.5:
                label = 1
                break
        
        win = data[:, start:end]
        med = np.median(win, axis=1, keepdims=True)
        mad = np.median(np.abs(win - med), axis=1, keepdims=True) + 1e-6
        win = (win - med) / mad
        
        if n_chans >= N_CHANS_OUT:
            idx = np.sort(rng_file.choice(n_chans, N_CHANS_OUT, replace=False))
            win = win[idx]
        else:
            pad = np.zeros((N_CHANS_OUT - n_chans, N_TIMES), dtype=win.dtype)
            win = np.vstack([win, pad])
        
        windows.append((win.astype(np.float32), label))
    
    return windows

# 7. Dataset class
class IctalDataset(Dataset):
    def __init__(self, subject_paths, dataset_name=""):
        self.samples = []
        print(f"Loading {dataset_name} dataset...")
        for sub_path in tqdm(subject_paths, desc=f"Processing {dataset_name} subjects"):
            is_omni = "omni" in str(sub_path).lower()
            if is_omni:
                ieeg_dir = sub_path / "ses-01" / "ieeg"
            else:
                ieeg_dir = sub_path / "ses-presurgery" / "ieeg"
            if not ieeg_dir.exists():
                ieeg_dir = sub_path / "ieeg"
                if not ieeg_dir.exists():
                    continue
            
            file_pattern = "*_ieeg.edf" if is_omni else "*_ieeg.vhdr"
            for ieeg_file in sorted(ieeg_dir.glob(file_pattern)):
                file_seed = SEED + hash(str(ieeg_file.absolute())) % 1000000
                
                if is_omni:
                    if "task-ictal" in ieeg_file.name:
                        ictal_intervals = [(0, 1e9)]
                    elif "task-interictal" in ieeg_file.name:
                        ictal_intervals = []
                    else:
                        print(f"Unknown task in {ieeg_file.name}")
                        continue
                else:
                    ev_file = Path(str(ieeg_file).replace("_ieeg.vhdr", "_events.tsv"))
                    ictal_intervals = parse_seizure_intervals(ev_file)
                
                try:
                    raw = load_run(ieeg_file)
                except Exception as e:
                    print(f"Skip {sub_path.name}/{ieeg_file.name}: {e}")
                    continue
                
                wins = extract_windows(raw.get_data(), TARGET_SFREQ, ictal_intervals, file_seed)
                self.samples.extend(wins)

        n_ictal = sum(l for _, l in self.samples)
        n_inter = len(self.samples) - n_ictal
        print(f"\n  {dataset_name} TOTAL: {len(self.samples)} | "
              f"Ictal: {n_ictal} ({100*n_ictal/max(len(self.samples),1):.1f}%) | "
              f"Interictal: {n_inter}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

# 8. Model loading (LUNA con adattamento posizionale)
KEY_MAP = {
    "cross_attn.temparature": "cross_attn.temperature",
    "channel_location_embedder.0.fc1.weight": "channel_location_embedder.fc1.weight",
    "channel_location_embedder.0.fc1.bias":   "channel_location_embedder.fc1.bias",
    "channel_location_embedder.0.fc2.weight": "channel_location_embedder.fc2.weight",
    "channel_location_embedder.0.fc2.bias":   "channel_location_embedder.fc2.bias",
}
SKIP_KEYS = {
    "channel_emb.embeddings.weight",
    "channel_location_embedder.0.norm.weight", "channel_location_embedder.0.norm.bias",
    "cross_attn.ffn.norm.weight", "cross_attn.ffn.norm.bias",
    "decoder_head.decoder_linear.fc1.bias", "decoder_head.decoder_linear.fc1.weight",
    "decoder_head.decoder_linear.fc2.bias", "decoder_head.decoder_linear.fc2.weight",
    "decoder_head.decoder_pred.layers.0.linear1.bias", "decoder_head.decoder_pred.layers.0.linear1.weight",
    "decoder_head.decoder_pred.layers.0.linear2.bias", "decoder_head.decoder_pred.layers.0.linear2.weight",
    "decoder_head.decoder_pred.layers.0.multihead_attn.in_proj_bias", "decoder_head.decoder_pred.layers.0.multihead_attn.in_proj_weight",
    "decoder_head.decoder_pred.layers.0.multihead_attn.out_proj.bias", "decoder_head.decoder_pred.layers.0.multihead_attn.out_proj.weight",
    "decoder_head.decoder_pred.layers.0.norm1.bias", "decoder_head.decoder_pred.layers.0.norm1.weight",
    "decoder_head.decoder_pred.layers.0.norm2.bias", "decoder_head.decoder_pred.layers.0.norm2.weight",
    "decoder_head.decoder_pred.layers.0.norm3.bias", "decoder_head.decoder_pred.layers.0.norm3.weight",
    "decoder_head.decoder_pred.layers.0.self_attn.in_proj_bias", "decoder_head.decoder_pred.layers.0.self_attn.in_proj_weight",
    "decoder_head.decoder_pred.layers.0.self_attn.out_proj.bias", "decoder_head.decoder_pred.layers.0.self_attn.out_proj.weight",
    "decoder_head.norm.bias", "decoder_head.norm.weight",
}

def load_luna():
    """Carica LUNA con interpolazione degli embedding posizionali se N_TIMES cambia."""
    model = LUNA(n_outputs=2, n_chans=64, n_times=N_TIMES,
                 embed_dim=64, num_queries=4, depth=8)
    raw_sd = load_file(str(CKPT_PATH), device="cpu")
    
    sd = {}
    old_len = 2000  # lunghezza originale del checkpoint pre-addestrato
    
    for k, v in raw_sd.items():
        if k in SKIP_KEYS:
            continue
        new_k = KEY_MAP.get(k, k)
        
        # Interpola embedding posizionali/sequence-dependent se la dimensione temporale cambia
        if v.dim() >= 2 and v.shape[-1] == old_len and old_len != N_TIMES:
            v_interp = torch.nn.functional.interpolate(
                v.unsqueeze(0) if v.dim() == 2 else v, 
                size=N_TIMES, mode='linear', align_corners=False
            ).squeeze(0)
            sd[new_k] = v_interp
            print(f"  ⚠ Interpolated {k}: {old_len} → {N_TIMES}")
        else:
            sd[new_k] = v
            
    model.load_state_dict(sd, strict=False)
    print(f"✓ Pre-trained weights loaded (adapted for n_times={N_TIMES})")
    return model

model = load_luna().to(DEVICE)

# 9. Load datasets
print("LOADING TRAINING DATASET")
train_ds = IctalDataset(train_paths, dataset_name="TRAIN")
print("\nLOADING VALIDATION DATASET")
val_ds = IctalDataset(val_paths, dataset_name="VAL")

# Class weights
labels = [train_ds[i][1].item() for i in range(len(train_ds))]
n_ictal = sum(labels)
n_inter = len(labels) - n_ictal
n_total = n_ictal + n_inter

class_weight_0 = n_total / (2 * n_inter) if n_inter > 0 else 1.0
class_weight_1 = n_total / (2 * n_ictal) if n_ictal > 0 else 1.0

print(f"Class distribution - Interictal: {n_inter}, Ictal: {n_ictal}")
print(f"Class weights - class 0: {class_weight_0:.3f}, class 1: {class_weight_1:.3f}\n")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 10. Loss and optimizer
class_weights = torch.tensor([class_weight_0, class_weight_1]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

head_params = list(model.final_layer.parameters())
backbone_params = [p for n, p in model.named_parameters() if not n.startswith("final_layer")]

def get_optimizer(freeze_backbone: bool):
    for p in backbone_params:
        p.requires_grad = not freeze_backbone
    if freeze_backbone:
        return torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=1e-4)
    else:
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ], weight_decay=1e-4)

optimizer = get_optimizer(freeze_backbone=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# CSV logging locally
metrics_file = SAVE_DIR / "training_metrics.csv"
metrics_file.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_f1", "val_prec", "val_rec", "val_f1"])

# 11. Training loop con Gradient Accumulation
best_val_f1 = 0.0
patience_counter = 0

print("STARTING TRAINING")
for epoch in range(1, N_EPOCHS + 1):
    if epoch == FROZEN_EPOCHS + 1:
        print(f"\n>>> Epoch {epoch}: unfreezing backbone")
        optimizer = get_optimizer(freeze_backbone=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS - FROZEN_EPOCHS
        )
    
    # Training phase
    model.train()
    total_loss, tp, fp, fn = 0.0, 0, 0, 0
    optimizer.zero_grad()  # Reset all'inizio dell'epoch
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
    for step, (X, y) in enumerate(train_loop):
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # Forward + loss scalata per accumulo
        logits = model(X)
        loss = criterion(logits, y) / GRAD_ACCUM_STEPS
        loss.backward()
        
        # Step ottimizzatore solo ogni N accumuli o all'ultimo batch
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        preds = logits.argmax(dim=1)
        # Logging: moltiplico per GRAD_ACCUM_STEPS per avere la loss "reale"
        total_loss += (loss.item() * GRAD_ACCUM_STEPS) * len(y)
        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()
        train_loop.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)
    
    avg_loss = total_loss / len(train_ds)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    train_f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    scheduler.step()
    
    # Validation phase
    model.eval()
    v_tp, v_fp, v_fn = 0, 0, 0
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(dim=1)
            v_tp += ((preds == 1) & (y == 1)).sum().item()
            v_fp += ((preds == 1) & (y == 0)).sum().item()
            v_fn += ((preds == 0) & (y == 1)).sum().item()
    
    v_prec = v_tp / max(v_tp + v_fp, 1)
    v_rec = v_tp / max(v_tp + v_fn, 1)
    val_f1 = 2 * v_prec * v_rec / max(v_prec + v_rec, 1e-6)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "train/loss": avg_loss,
        "train/precision": prec,
        "train/recall": rec,
        "train/f1": train_f1,
        "val/precision": v_prec,
        "val/recall": v_rec,
        "val/f1": val_f1,
        "lr_head": LR_HEAD if epoch <= FROZEN_EPOCHS else LR_HEAD,
        "lr_backbone": 0 if epoch <= FROZEN_EPOCHS else LR_BACKBONE,
        "config/grad_accum_steps": GRAD_ACCUM_STEPS,
        "config/window_sec": WINDOW_SEC,
    })
    
    # Save to local CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, train_f1, v_prec, v_rec, val_f1])
    
    print(f"Epoch {epoch:2d} | Tr.loss={avg_loss:.4f} F1={train_f1:.3f} | "
          f"Val prec={v_prec:.3f} rec={v_rec:.3f} F1={val_f1:.3f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), SAVE_DIR / "best.pt")
        print(f"  ✓ Saved (val F1={val_f1:.4f})")
        
        if config["wandb"].get("log_model", True):
            artifact = wandb.Artifact(
                name=f"luna_ieeg_best",
                type="model",
                metadata={"val_f1": val_f1, "epoch": epoch}
            )
            artifact.add_file(SAVE_DIR / "best.pt")
            wandb.log_artifact(artifact)
        
        patience_counter = 0
    else:
        patience_counter += 1
        if EARLY_STOPPING_PATIENCE and patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs")
            break

print(f"\n\nTraining completed. Best val F1: {best_val_f1:.4f}")
print(f"Model saved to: {SAVE_DIR / 'best.pt'}")
print(f"Training metrics saved to: {metrics_file}")
wandb.finish()