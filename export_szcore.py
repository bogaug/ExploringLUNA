"""
export_szcore_hup.py
Run LUNA inference on the HUP hold-out test subjects and export reference/
hypothesis annotations in the epilepsy2bids/SzCORE format, to be scored with:

    python -m evaluate ref_dataset hyp_dataset

Writes two folder trees with the same structure (one .tsv per recording):
    ref_dataset/<sub-HUPxxx>/<recording>.tsv   (true seizures)
    hyp_dataset/<sub-HUPxxx>/<recording>.tsv   (model-predicted seizures)

Preprocessing is identical to fine-tuning.py (notch + bandpass, z-score,
cyclic channel repetition) to avoid any train/inference mismatch.
HUP only - no SWEC.
"""

import re
import hashlib
import datetime
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn.functional as F
from pathlib import Path
from braindecode.models import LUNA
from tqdm import tqdm
import yaml
import epilepsy2bids.annotations as e2b

# ------------------------------ Config ------------------------------
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

BANDPASS_LOW = config.get("bandpass_low", 0.1)
BANDPASS_HIGH = config.get("bandpass_high", 75.0)
NOTCH_FREQ = config.get("notch_freq", 50.0)
MAX_SEIZURE_DUR = config.get("max_seizure_dur", 300.0)
DEFAULT_SEIZURE_DUR = config.get("default_seizure_dur", 120.0)
MERGE_GAP_SEC = config.get("merge_gap_sec", 10.0)

REF_DIR = Path("./ref_dataset")
HYP_DIR = Path("./hyp_dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ Model ------------------------------
def load_model():
    model = LUNA(n_outputs=2, n_chans=N_CHANS_OUT, n_times=N_TIMES,
                 embed_dim=64, num_queries=4, depth=8)
    sd = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model.to(DEVICE)


# ------------------------------ Event parsing (same as training) ------------------------------
OFFSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*(?:offset|end)"
    r"|\boffset\b|seizure\s*off|devolution|\bend\b", re.I)
ONSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*onset"
    r"|(?<!off)\bonset\b|^sz$|sz\s*event", re.I)

def classify_marker(label: str):
    l = str(label).strip().lower()
    if OFFSET_RE.search(l):
        return "OFFSET"
    if ONSET_RE.search(l):
        return "ONSET"
    return None

def parse_seizure_intervals(events_tsv: Path):
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
    intervals, pending = [], None
    for t, kind in markers:
        if kind == "ONSET":
            if pending is not None:
                intervals.append((pending, min(pending + DEFAULT_SEIZURE_DUR, t)))
            pending = t
        else:
            if pending is not None:
                dur = t - pending
                intervals.append((pending, pending + DEFAULT_SEIZURE_DUR)
                                 if (dur <= 0 or dur > MAX_SEIZURE_DUR) else (pending, t))
                pending = None
    if pending is not None:
        intervals.append((pending, pending + DEFAULT_SEIZURE_DUR))
    return intervals


# ------------------------------ Loading + preprocessing (same as training) ------------------------------
def load_run(data_path: Path):
    if not data_path.resolve().exists():
        raise FileNotFoundError(data_path.name)
    suffix = data_path.suffix.lower()
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(data_path), preload=True, verbose=False)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(data_path), preload=True, verbose=False)
    else:
        raise OSError(f"Unsupported extension: {data_path.name}")
    ch_file = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_channels.tsv", str(data_path)))
    if ch_file.exists():
        ch_df = pd.read_csv(ch_file, sep="\t")
        if "status" in ch_df.columns:
            good = ch_df[ch_df["status"] == "good"]["name"].tolist()
            available = [c for c in good if c in raw.ch_names]
            if available:
                raw.pick(available)
    sfreq_now = raw.info["sfreq"]; n_samples = raw.n_times
    if n_samples > int(3.3 * sfreq_now / 1.0):
        try:
            raw.notch_filter(freqs=[NOTCH_FREQ], verbose=False)
        except Exception:
            pass
    if n_samples > int(3.3 * sfreq_now / BANDPASS_LOW):
        nyq = sfreq_now / 2.0
        raw.filter(l_freq=BANDPASS_LOW, h_freq=min(BANDPASS_HIGH, nyq - 1.0), verbose=False)
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1:
        raw.resample(TARGET_SFREQ, verbose=False)
    return raw

def stable_seed(path: Path) -> int:
    h = hashlib.md5(str(path.absolute()).encode()).hexdigest()
    return (SEED + int(h, 16)) % (2**32)


# ------------------------------ Prediction -> intervals (seconds) ------------------------------
def predict_intervals(model, data_path: Path):
    raw = load_run(data_path)
    data = raw.get_data()
    n_chans, total = data.shape
    total_sec = total / TARGET_SFREQ
    step = int(N_TIMES * (1 - OVERLAP))

    rng = np.random.default_rng(stable_seed(data_path))
    if n_chans >= N_CHANS_OUT:
        chan_idx = np.sort(rng.choice(n_chans, N_CHANS_OUT, replace=False))
    else:
        reps = int(np.ceil(N_CHANS_OUT / n_chans))
        chan_idx = np.tile(np.arange(n_chans), reps)[:N_CHANS_OUT]

    win_starts, preds = [], []
    for start in range(0, total - N_TIMES + 1, step):
        win = data[chan_idx, start:start + N_TIMES]
        mean = win.mean(axis=1, keepdims=True)
        std = win.std(axis=1, keepdims=True) + 1e-6
        win = (win - mean) / std
        x = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)[0, 1].item()
        win_starts.append(start / TARGET_SFREQ)
        preds.append(int(prob > PROB_THRESH))

    intervals = []
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            s = win_starts[i]; j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            e = win_starts[j - 1] + WINDOW_SEC
            intervals.append((s, e))
            i = j
        else:
            i += 1

    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < MERGE_GAP_SEC:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged, total_sec


# ------------------------------ Write annotations (epilepsy2bids format) ------------------------------
def write_annotations(intervals, total_sec, out_tsv: Path):
    start_dt = datetime.datetime(2000, 1, 1)
    anns = e2b.Annotations()
    if len(intervals) == 0:
        a = e2b.Annotation()
        a["onset"] = 0; a["duration"] = total_sec; a["eventType"] = e2b.EventType.bckg
        a["confidence"] = "n/a"; a["channels"] = "n/a"
        a["dateTime"] = start_dt; a["recordingDuration"] = total_sec
        anns.events.append(a)
    else:
        for (onset, offset) in intervals:
            a = e2b.Annotation()
            a["onset"] = float(onset); a["duration"] = float(offset - onset)
            a["eventType"] = e2b.EventType.sz
            a["confidence"] = "n/a"; a["channels"] = "n/a"
            a["dateTime"] = start_dt; a["recordingDuration"] = total_sec
            anns.events.append(a)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    anns.saveTsv(str(out_tsv))


# ------------------------------ Main ------------------------------
def main():
    print(f"Loading model from {WEIGHTS_PATH}")
    model = load_model()

    n_files = 0
    for subj in TEST_SUBJECTS:
        subj_path = BIDS_ROOT / subj
        if not subj_path.exists():
            print(f"Subject not found, skipping: {subj}")
            continue
        ieeg_dir = subj_path / "ses-presurgery" / "ieeg"
        data_files = sorted(ieeg_dir.glob("*_ieeg.vhdr")) or sorted(ieeg_dir.glob("*_ieeg.edf"))
        for fpath in tqdm(data_files, desc=subj):
            ev_file = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_events.tsv", str(fpath)))
            try:
                true_intervals = parse_seizure_intervals(ev_file)
                pred_intervals, total_sec = predict_intervals(model, fpath)
            except Exception as e:
                tqdm.write(f"  Skip {fpath.name}: {e}")
                continue
            rel = f"{subj}/{fpath.stem}.tsv"   # subj already has the sub- prefix
            write_annotations(true_intervals, total_sec, REF_DIR / rel)
            write_annotations(pred_intervals, total_sec, HYP_DIR / rel)
            n_files += 1

    print(f"\nExported {n_files} files to:")
    print(f"    {REF_DIR.resolve()}")
    print(f"    {HYP_DIR.resolve()}")
    print(f"\nScore with SzCORE:")
    print(f"    python -m evaluate {REF_DIR} {HYP_DIR}")


if __name__ == "__main__":
    main()