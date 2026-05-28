"""
Inference script for LUNA model on held-out test subjects (ds003029).
Logs results to Weights & Biases: bar chart of metrics and confusion matrix.
"""

import json
import re
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn.functional as F
from pathlib import Path
from braindecode.models import LUNA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import yaml
import wandb

# ------------------------------
# Load configuration
# ------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BIDS_ROOT = Path("../dataset/ds003029")
WEIGHTS_PATH = Path(config["save_dir"]) / "best.pt"
TARGET_SFREQ = config["target_sfreq"]
WINDOW_SEC = config["window_sec"]
N_TIMES = int(TARGET_SFREQ * WINDOW_SEC)
N_CHANS_OUT = config["n_chans_out"]
OVERLAP = config["overlap"]
SEED = config["seed"]
PROB_THRESH = config["prob_threshold"]
TEST_SUBJECTS = config["test_subjects"]
DEVICE = "cuda" 

# ------------------------------
# Load model
# ------------------------------
def load_model():
    model = LUNA(n_outputs=2, n_chans=64, n_times=2000,
                 embed_dim=64, num_queries=4, depth=8)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)

print(f"Loading model from {WEIGHTS_PATH}")
model = load_model()

# ------------------------------
# Preprocessing, parsing, event metrics (identical to training)
# ------------------------------
def load_run(vhdr_path: Path):
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    ch_file = Path(str(vhdr_path).replace("_ieeg.vhdr", "_channels.tsv"))
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

ONSET_RE = re.compile(r"sz event|seizure.*onset|ictal onset|\bonset\b|\bsz\b|poss.*onset|clinical onset", re.I)
OFFSET_RE = re.compile(r"electrographic end|offset|sz end|seizure off|devolution|ending|clinical end", re.I)

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

def compute_event_metrics(intervals_true, pred_windows_tstart, pred_labels, total_duration_sec, window_len_sec=WINDOW_SEC, overlap=OVERLAP):
    # Build candidate events (consecutive ictal windows)
    pred_intervals = []
    i = 0
    while i < len(pred_labels):
        if pred_labels[i] == 1:
            start = pred_windows_tstart[i]
            j = i
            while j < len(pred_labels) and pred_labels[j] == 1:
                j += 1
            end = pred_windows_tstart[j-1] + window_len_sec
            pred_intervals.append((start, end))
            i = j
        else:
            i += 1
    
    detected = 0
    for on, off in intervals_true:
        for ps, pe in pred_intervals:
            if max(ps, on) < min(pe, off):
                detected += 1
                break
    sensitivity = detected / max(len(intervals_true), 1)
    
    fp_events = 0
    for ps, pe in pred_intervals:
        overlap = False
        for on, off in intervals_true:
            if max(ps, on) < min(pe, off):
                overlap = True
                break
        if not overlap:
            fp_events += 1
    
    ictal_duration = sum(off - on for on, off in intervals_true)
    interictal_hours = max((total_duration_sec - ictal_duration) / 3600.0, 1e-6)
    fp_per_hour = fp_events / interictal_hours
    
    latencies = []
    for on, off in intervals_true:
        first_pred_start = None
        for ps, pe in sorted(pred_intervals, key=lambda x: x[0]):
            if ps < off and pe > on:
                first_pred_start = ps
                break
        if first_pred_start is not None:
            latencies.append(first_pred_start - on)
    avg_latency = np.mean(latencies) if latencies else float('inf')
    
    return {
        "sensitivity": sensitivity,
        "false_positives_per_hour": fp_per_hour,
        "avg_latency_sec": avg_latency,
        "n_true_events": len(intervals_true),
        "n_detected": detected,
        "n_false_positive_events": fp_events
    }

# ------------------------------
# Inference on a single subject (global RNG for channel selection)
# ------------------------------
def run_inference(subject: str):
    ieeg_dir = BIDS_ROOT / subject / "ses-presurgery" / "ieeg"
    results = []
    vhdr_files = sorted(ieeg_dir.glob("*_ieeg.vhdr"))
    rng_global = np.random.default_rng(SEED)   # same as training (global seed)
    
    for vhdr in tqdm(vhdr_files, desc=f"Inference on {subject}"):
        ev_file = Path(str(vhdr).replace("_ieeg.vhdr", "_events.tsv"))
        intervals = parse_seizure_intervals(ev_file)
        raw = load_run(vhdr)
        data = raw.get_data()
        n_chans, total = data.shape
        step = int(N_TIMES * (1 - OVERLAP))
        
        t_axis, y_true, y_pred, y_prob = [], [], [], []
        
        for start in range(0, total - N_TIMES, step):
            end = start + N_TIMES
            t_start = start / TARGET_SFREQ
            t_end = end / TARGET_SFREQ
            
            # Ground truth label (window-level)
            label = 0
            for (on, off) in intervals:
                overlap = max(0, min(t_end, off) - max(t_start, on))
                if overlap / (t_end - t_start) > 0.5:
                    label = 1
                    break
            
            win = data[:, start:end]
            med = np.median(win, axis=1, keepdims=True)
            mad = np.median(np.abs(win - med), axis=1, keepdims=True) + 1e-6
            win = (win - med) / mad
            
            # Channel adaptation (global RNG, same as training)
            if n_chans >= N_CHANS_OUT:
                idx = np.sort(rng_global.choice(n_chans, N_CHANS_OUT, replace=False))
                win = win[idx]
            else:
                pad = np.zeros((N_CHANS_OUT - n_chans, N_TIMES))
                win = np.vstack([win, pad])
            
            x = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                prob = F.softmax(logits, dim=1)[0, 1].item()
                pred = int(prob > PROB_THRESH)
            
            t_axis.append(t_start)
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)
        
        # Window-level metrics for this run
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        tp = ((y_pred_np == 1) & (y_true_np == 1)).sum()
        fp = ((y_pred_np == 1) & (y_true_np == 0)).sum()
        fn = ((y_pred_np == 0) & (y_true_np == 1)).sum()
        tn = ((y_pred_np == 0) & (y_true_np == 0)).sum()
        
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        
        total_duration = total / TARGET_SFREQ
        event_metrics = compute_event_metrics(intervals, t_axis, y_pred, total_duration)
        
        run_name = vhdr.stem.split("_run-")[1].split("_")[0]
        print(f"  run-{run_name} | windows={len(y_true)} | ictal_gt={y_true_np.sum()} "
              f"| prec={prec:.3f} | rec={rec:.3f} | F1={f1:.3f} | acc={acc:.3f}")
        print(f"        Event metrics: sens={event_metrics['sensitivity']:.2f} "
              f"FP/hour={event_metrics['false_positives_per_hour']:.2f} "
              f"latency={event_metrics['avg_latency_sec']:.1f}s")
        
        results.append({
            "run": run_name,
            "t": t_axis,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "window_metrics": {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc},
            "event_metrics": event_metrics
        })
    return results

# ------------------------------
# Run inference on all test subjects and log to wandb
# ------------------------------
wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=f"eval_{config['wandb'].get('run_name', 'inference')}",
    tags=["inference", "test-set"] + config["wandb"].get("tags", [])
)

all_results = {}
all_y_true_global = []
all_y_pred_global = []

for subj in TEST_SUBJECTS:
    if not (BIDS_ROOT / subj).exists():
        print(f"\n=== {subj} NOT FOUND, skipping ===")
        continue
    print(f"\n=== Inference on {subj} ===")
    results = run_inference(subj)
    all_results[subj] = results
    for r in results:
        if len(r["y_true"]) > 0:
            all_y_true_global.extend(r["y_true"])
            all_y_pred_global.extend(r["y_pred"])

# Global window-level metrics
if all_y_true_global:
    yt = np.array(all_y_true_global)
    yp = np.array(all_y_pred_global)
    tp = ((yp == 1) & (yt == 1)).sum()
    fp = ((yp == 1) & (yt == 0)).sum()
    tn = ((yp == 0) & (yt == 0)).sum()
    fn = ((yp == 0) & (yt == 1)).sum()
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    
    print("\n" + "="*60)
    print("GLOBAL WINDOW-LEVEL METRICS (All test subjects combined)")
    print("="*60)
    print(f"  Total windows:    {len(yt)}")
    print(f"  True Positives:   {tp}")
    print(f"  False Positives:  {fp}")
    print(f"  True Negatives:   {tn}")
    print(f"  False Negatives:  {fn}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Accuracy:  {acc:.3f}")
    print("="*60)
    
    # Log scalars to wandb
    wandb.log({
        "test/total_windows": len(yt),
        "test/precision": prec,
        "test/recall": rec,
        "test/f1": f1,
        "test/accuracy": acc,
        "test/true_positives": tp,
        "test/false_positives": fp,
        "test/true_negatives": tn,
        "test/false_negatives": fn,
    })
    
    # Aggregate event metrics across runs (optional)
    all_sens = []
    all_fp_per_hour = []
    all_latency = []
    for subj, reslist in all_results.items():
        for r in reslist:
            em = r["event_metrics"]
            all_sens.append(em["sensitivity"])
            all_fp_per_hour.append(em["false_positives_per_hour"])
            if em["avg_latency_sec"] != float('inf'):
                all_latency.append(em["avg_latency_sec"])
    avg_sens = np.mean(all_sens) if all_sens else 0
    avg_fp_rate = np.mean(all_fp_per_hour) if all_fp_per_hour else 0
    avg_lat = np.mean(all_latency) if all_latency else float('inf')
    wandb.log({
        "test/avg_event_sensitivity": avg_sens,
        "test/avg_fp_events_per_hour": avg_fp_rate,
        "test/avg_detection_latency_sec": avg_lat
    })
    
    # --- Figure 1: Confusion Matrix ---
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Interictal', 'Ictal'],
                yticklabels=['Interictal', 'Ictal'],
                annot_kws={'size': 16}, ax=ax_cm, cbar=False)
    ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax_cm.set_ylabel('True Label', fontsize=12)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    wandb.log({"test/confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)
    
    # --- Figure 2: Bar chart of metrics ---
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    metrics_values = [prec, rec, f1, acc]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    bars = ax_bar.bar(metrics_names, metrics_values, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, metrics_values):
        ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Score', fontsize=12)
    ax_bar.set_title('Aggregate Metrics on Test Set', fontsize=14, fontweight='bold')
    ax_bar.set_ylim(0, 1.15)
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    wandb.log({"test/metrics_bar_chart": wandb.Image(fig_bar)})
    plt.close(fig_bar)
    
    # Also save figures locally (optional)
    fig_cm.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    fig_bar.savefig('metrics_barchart.png', dpi=150, bbox_inches='tight')
    print("\n✓ Figures saved locally: confusion_matrix.png, metrics_barchart.png")

# Save detailed results to JSON
out = {}
for subj, results in all_results.items():
    out[subj] = []
    for r in results:
        out[subj].append({
            "run": r["run"],
            "t": r["t"],
            "y_true": r["y_true"],
            "y_pred": r["y_pred"],
            "y_prob": r["y_prob"],
            "window_metrics": r["window_metrics"],
            "event_metrics": r["event_metrics"]
        })
with open("inference_results_improved.json", "w") as f:
    json.dump(out, f, indent=2)
print("✓ Results saved to inference_results_improved.json")

wandb.finish()