"""
Inference script for LUNA model on held-out test subjects.
ALLINEATO a fine-tuning.py: stesso parser, stesso preprocessing (notch+bandpass,
z-score), stessa gestione canali (ripetizione ciclica, NO zero-padding), seed
deterministico. Logga su W&B: bar chart, confusion matrix, artifact.

Aggiunge: ricerca della soglia ottimale sulle probabilità (oltre alla soglia fissa
del config) per diagnosticare se la bassa precision dipende dalla soglia 0.5.
"""

import json
import re
import hashlib
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
# Config
# ------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BIDS_ROOT = Path("./dataset/ds004100")
WEIGHTS_PATH = Path(config["save_dir"]) / "best.pt"
TARGET_SFREQ = config["target_sfreq"]
WINDOW_SEC = config["window_sec"]
N_TIMES = int(TARGET_SFREQ * WINDOW_SEC)
N_CHANS_OUT = config["n_chans_out"]
OVERLAP = config["overlap"]
SEED = config["seed"]
PROB_THRESH = config["prob_threshold"]
TEST_SUBJECTS = config["test_subjects"]

# Preprocessing: STESSI default del fine-tuning
BANDPASS_LOW = config.get("bandpass_low", 0.1)
BANDPASS_HIGH = config.get("bandpass_high", 75.0)
NOTCH_FREQ = config.get("notch_freq", 50.0)
MAX_SEIZURE_DUR = config.get("max_seizure_dur", 300.0)
DEFAULT_SEIZURE_DUR = config.get("default_seizure_dur", 120.0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Model
# ------------------------------
def load_model():
    # n_times deve combaciare con il training (1280), non 2000
    model = LUNA(n_outputs=2, n_chans=N_CHANS_OUT, n_times=N_TIMES,
                 embed_dim=64, num_queries=4, depth=8)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)

print(f"Loading model from {WEIGHTS_PATH}")
model = load_model()

# ------------------------------
# Parser eventi — IDENTICO al fine-tuning (OFFSET prima di ONSET)
# ------------------------------
OFFSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*(?:offset|end)"
    r"|\boffset\b|seizure\s*off|devolution|\bend\b",
    re.I,
)
ONSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*onset"
    r"|(?<!off)\bonset\b|^sz$|sz\s*event",
    re.I,
)

def classify_marker(label: str):
    l = str(label).strip().lower()
    if OFFSET_RE.search(l):
        return "OFFSET"
    if ONSET_RE.search(l):
        return "ONSET"
    return None

def parse_seizure_intervals(events_tsv: Path,
                            max_dur: float = MAX_SEIZURE_DUR,
                            default_dur: float = DEFAULT_SEIZURE_DUR):
    if not events_tsv.exists():
        return []
    df = pd.read_csv(events_tsv, sep="\t")
    if "onset" not in df.columns or "trial_type" not in df.columns:
        return []
    markers = []
    for _, row in df.iterrows():
        kind = classify_marker(row.get("trial_type", ""))
        if kind is None:
            continue
        try:
            t = float(row["onset"])
        except (ValueError, TypeError):
            continue
        markers.append((t, kind))
    markers.sort(key=lambda x: x[0])
    intervals = []
    pending_onset = None
    for t, kind in markers:
        if kind == "ONSET":
            if pending_onset is not None:
                intervals.append((pending_onset, min(pending_onset + default_dur, t)))
            pending_onset = t
        else:
            if pending_onset is not None:
                dur = t - pending_onset
                if dur <= 0 or dur > max_dur:
                    intervals.append((pending_onset, pending_onset + default_dur))
                else:
                    intervals.append((pending_onset, t))
                pending_onset = None
    if pending_onset is not None:
        intervals.append((pending_onset, pending_onset + default_dur))
    return intervals

# ------------------------------
# Caricamento + PREPROCESSING — IDENTICO al fine-tuning
# ------------------------------
def load_run(data_path: Path):
    if not data_path.resolve().exists():
        raise FileNotFoundError(f"Unfetched/missing data file: {data_path.name}")
    suffix = data_path.suffix.lower()
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(data_path), preload=True, verbose=False)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(data_path), preload=True, verbose=False)
    else:
        raise OSError(f"Estensione non supportata: {data_path.name}")

    ch_file = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_channels.tsv", str(data_path)))
    if ch_file.exists():
        ch_df = pd.read_csv(ch_file, sep="\t")
        if "status" in ch_df.columns:
            good = ch_df[ch_df["status"] == "good"]["name"].tolist()
            available = [c for c in good if c in raw.ch_names]
            if available:
                raw.pick(available)

    # Stesso ordine del training: notch -> bandpass -> resample, con guardie
    sfreq_now = raw.info["sfreq"]
    n_samples = raw.n_times
    min_len_for_filter = int(3.3 * sfreq_now / BANDPASS_LOW)
    can_filter = n_samples > min_len_for_filter

    if n_samples > int(3.3 * sfreq_now / 1.0):
        try:
            raw.notch_filter(freqs=[NOTCH_FREQ], verbose=False)
        except Exception as e:
            print(f"  ⚠ notch saltato su {data_path.name}: {e}")
    if can_filter:
        nyq = sfreq_now / 2.0
        h_freq = min(BANDPASS_HIGH, nyq - 1.0)
        raw.filter(l_freq=BANDPASS_LOW, h_freq=h_freq, verbose=False)
    else:
        print(f"  ⚠ Filtraggio saltato su {data_path.name}: troppo corto ({n_samples})")
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1:
        raw.resample(TARGET_SFREQ, verbose=False)
    return raw

def stable_seed(path: Path) -> int:
    h = hashlib.md5(str(path.absolute()).encode()).hexdigest()
    return (SEED + int(h, 16)) % (2**32)

# ------------------------------
# Event metrics (invariato)
# ------------------------------
def compute_event_metrics(intervals_true, pred_windows_tstart, pred_labels,
                          total_duration_sec, window_len_sec=WINDOW_SEC, overlap=OVERLAP):
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
        ov = False
        for on, off in intervals_true:
            if max(ps, on) < min(pe, off):
                ov = True
                break
        if not ov:
            fp_events += 1
    ictal_duration = sum(off - on for on, off in intervals_true)
    interictal_hours = max((total_duration_sec - ictal_duration) / 3600.0, 1e-6)
    fp_per_hour = fp_events / interictal_hours
    return {
        "sensitivity": sensitivity,
        "false_positives_per_hour": fp_per_hour,
        "n_true_events": len(intervals_true),
        "n_detected": detected,
        "n_false_positive_events": fp_events,
    }

# ------------------------------
# Inference su un soggetto
# ------------------------------
def run_inference(subject: str):
    ieeg_dir = BIDS_ROOT / subject / "ses-presurgery" / "ieeg"
    results = []
    data_files = sorted(ieeg_dir.glob("*_ieeg.vhdr"))
    if not data_files:
        data_files = sorted(ieeg_dir.glob("*_ieeg.edf"))

    for fpath in tqdm(data_files, desc=f"Inference on {subject}"):
        ev_file = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_events.tsv", str(fpath)))
        intervals = parse_seizure_intervals(ev_file)
        try:
            raw = load_run(fpath)
        except (FileNotFoundError, OSError) as e:
            tqdm.write(f"⚠ Skipping {fpath.name}: {e}")
            continue
        data = raw.get_data()
        n_chans, total = data.shape
        step = int(N_TIMES * (1 - OVERLAP))

        # Selezione canali UNA volta per file, deterministica (come training)
        rng_file = np.random.default_rng(stable_seed(fpath))
        if n_chans >= N_CHANS_OUT:
            chan_idx = np.sort(rng_file.choice(n_chans, N_CHANS_OUT, replace=False))
        else:
            reps = int(np.ceil(N_CHANS_OUT / n_chans))
            chan_idx = np.tile(np.arange(n_chans), reps)[:N_CHANS_OUT]

        t_axis, y_true, y_pred, y_prob = [], [], [], []
        for start in range(0, total - N_TIMES + 1, step):
            end = start + N_TIMES
            t_start = start / TARGET_SFREQ
            t_end = end / TARGET_SFREQ

            label = 0
            for (on, off) in intervals:
                ov = max(0, min(t_end, off) - max(t_start, on))
                if ov / (t_end - t_start) > 0.5:
                    label = 1
                    break

            win = data[chan_idx, start:end]
            # z-score per canale (IDENTICO al training, NON median/MAD)
            mean = win.mean(axis=1, keepdims=True)
            std = win.std(axis=1, keepdims=True) + 1e-6
            win = (win - mean) / std

            x = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                prob = F.softmax(logits, dim=1)[0, 1].item()
                pred = int(prob > PROB_THRESH)

            t_axis.append(t_start)
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

        results.append({
            "run": fpath.stem,
            "t": t_axis, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
            "total_duration": total / TARGET_SFREQ,
            "intervals": intervals,
        })
    return results

# ------------------------------
# Esecuzione
# ------------------------------
wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=f"eval_{config['wandb'].get('run_name', 'inference')}",
    tags=["inference", "test-set"] + config["wandb"].get("tags", []),
)

all_results = {}
all_y_true, all_y_pred, all_y_prob = [], [], []
n_runs = 0
for subj in TEST_SUBJECTS:
    if not (BIDS_ROOT / subj).exists():
        print(f"⚠ Soggetto non trovato, salto: {subj}")
        continue
    res = run_inference(subj)
    all_results[subj] = res
    n_runs += len(res)
    for r in res:
        all_y_true.extend(r["y_true"])
        all_y_pred.extend(r["y_pred"])
        all_y_prob.extend(r["y_prob"])

print(f"\n--- Riepilogo: {n_runs} run processati, {len(all_y_true)} finestre ---")

if all_y_true:
    yt = np.array(all_y_true); yp = np.array(all_y_pred); ypr = np.array(all_y_prob)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    print(f"\n[Soglia {PROB_THRESH}]  prec={precision:.3f} rec={recall:.3f} f1={f1:.3f}")
    print(f"  Distribuzione vera: ictali={int(yt.sum())} ({100*yt.mean():.1f}%), background={int((yt==0).sum())}")

    wandb.log({
        "test/precision": precision, "test/recall": recall, "test/f1": f1,
    })

    # Figura
    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Performance - Test Subjects (aligned preprocessing)",
                 fontsize=18, fontweight="bold")
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                annot_kws={"size": 22}, ax=ax_cm,
                xticklabels=["Interictal", "Ictal"], yticklabels=["Interictal", "Ictal"])
    ax_cm.set_title("Confusion Matrix"); ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    names = ["Precision", "Recall", "F1", "Accuracy"]
    vals = [precision, recall, f1, accuracy]
    bars = ax_bar.bar(names, vals, color=["#1f9bf0", "#2ca02c", "#ff9e00", "#e8463a"])
    ax_bar.set_ylim(0, 1.1); ax_bar.set_title("Aggregate Metrics")
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = Path("inference_outputs"); out_dir.mkdir(exist_ok=True)
    global_png = out_dir / "global_results.png"
    cm_png = out_dir / "confusion_matrix.png"
    fig.savefig(global_png, dpi=120)
    fig_cm, ax_only = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_only)
    fig_cm.savefig(cm_png)
    wandb.log({"results/global_results": wandb.Image(fig),
               "results/confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig); plt.close(fig_cm)

    # JSON (senza i vettori lunghi per leggibilità)
    summary = {s: [{"run": r["run"],
                    "n_windows": len(r["y_true"]),
                    "n_ictal_true": int(sum(r["y_true"]))} for r in res]
               for s, res in all_results.items()}
    json_path = out_dir / "inference_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    artifact = wandb.Artifact(name="inference_results_data", type="results")
    artifact.add_file(str(json_path)); artifact.add_file(str(cm_png)); artifact.add_file(str(global_png))
    wandb.log_artifact(artifact)
    print(f"✓ Risultati salvati in {out_dir.resolve()}")
else:
    print("\n✘ 0 finestre processate: controlla i path/skipping sopra.")

wandb.finish()