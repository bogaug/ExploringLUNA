"""
electrode_map.py
Top-down (axial) brain map of intracranial electrodes, colored by how strongly
each electrode "sees" the seizure (ictal energy ratio).

Uses the real fsaverage coordinates from the BIDS *_electrodes.tsv (x, y, z).
Electrodes without coordinates (scalp 10-20, marked n/a) are skipped.
No 3D rendering / pyvista required: pure matplotlib, robust on headless/ARM.

Ictal energy ratio per electrode:
    power(signal during true seizure) / power(signal during background)
High ratio  -> electrode strongly involved (seizure focus / propagation) = warm color
Low ratio   -> electrode silent during the seizure = cool color

Usage:
    python electrode_map.py
"""

import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import yaml

# ------------------------------ Config ------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BIDS_ROOT = Path("./dataset/ds004100")
TARGET_SFREQ = config["target_sfreq"]
NOTCH_FREQ = config.get("notch_freq", 50.0)
BANDPASS_LOW = config.get("bandpass_low", 0.1)
BANDPASS_HIGH = config.get("bandpass_high", 75.0)
MAX_SEIZURE_DUR = config.get("max_seizure_dur", 300.0)
DEFAULT_SEIZURE_DUR = config.get("default_seizure_dur", 120.0)
TEST_SUBJECTS = config["test_subjects"]

SUBJECT = TEST_SUBJECTS[0]
OUT_PNG = "electrode_map.png"


# ------------------------------ Event parsing (same as training) ------------------------------
OFFSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*(?:offset|end)"
    r"|\boffset\b|seizure\s*off|devolution|\bend\b", re.I)
ONSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*onset"
    r"|(?<!off)\bonset\b|^sz$|sz\s*event", re.I)

def classify_marker(label):
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


def load_run(data_path: Path):
    suffix = data_path.suffix.lower()
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(data_path), preload=True, verbose=False)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(data_path), preload=True, verbose=False)
    else:
        raise OSError(f"Unsupported extension: {data_path.name}")
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


# ------------------------------ Load electrode coordinates ------------------------------
def load_electrodes(subject):
    ieeg_dir = BIDS_ROOT / subject / "ses-presurgery" / "ieeg"
    elec_files = list(ieeg_dir.glob("*_electrodes.tsv"))
    if not elec_files:
        raise FileNotFoundError("No *_electrodes.tsv found")
    df = pd.read_csv(elec_files[0], sep="\t")
    # keep only electrodes with valid numeric coordinates (drop scalp n/a)
    coords = {}
    for _, row in df.iterrows():
        try:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        except (ValueError, TypeError):
            continue
        if np.isfinite([x, y, z]).all():
            coords[str(row["name"])] = np.array([x, y, z])
    # auto-detect units: if typical magnitude < 1, it's meters -> convert to mm
    mags = np.array([np.abs(v).max() for v in coords.values()])
    if mags.size and np.median(mags) < 1.0:
        coords = {k: v * 1000.0 for k, v in coords.items()}  # m -> mm
    return coords


# ------------------------------ Ictal energy per electrode ------------------------------
def ictal_energy_ratio(raw, intervals, coords):
    """For each electrode with coordinates, ratio of ictal power to background power."""
    data = raw.get_data()
    ch_names = raw.ch_names
    sf = raw.info["sfreq"]
    total = data.shape[1]

    # build ictal mask
    ictal = np.zeros(total, dtype=bool)
    for on, off in intervals:
        a, b = int(on * sf), int(off * sf)
        ictal[max(0, a):min(total, b)] = True
    if ictal.sum() == 0 or (~ictal).sum() == 0:
        return {}

    ratios = {}
    for name in coords:
        if name not in ch_names:
            continue
        sig = data[ch_names.index(name)]
        p_ict = np.mean(sig[ictal] ** 2)
        p_bkg = np.mean(sig[~ictal] ** 2) + 1e-20
        ratios[name] = 10 * np.log10((p_ict / p_bkg) + 1e-12)  # dB
    return ratios


# ------------------------------ Plot ------------------------------
def main():
    print(f"Subject: {SUBJECT}")
    coords = load_electrodes(SUBJECT)
    print(f"Electrodes with coordinates: {len(coords)}")

    ieeg_dir = BIDS_ROOT / SUBJECT / "ses-presurgery" / "ieeg"
    files = sorted(ieeg_dir.glob("*_ieeg.vhdr")) or sorted(ieeg_dir.glob("*_ieeg.edf"))
    # pick a recording with a seizure
    chosen, intervals = None, []
    for f in files:
        ev = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_events.tsv", str(f)))
        iv = parse_seizure_intervals(ev)
        if iv:
            chosen, intervals = f, iv
            break
    if chosen is None:
        raise RuntimeError("No recording with a seizure found")
    print(f"Recording: {chosen.name}  seizures: {intervals}")

    raw = load_run(chosen)
    ratios = ictal_energy_ratio(raw, intervals, coords)
    print(f"Electrodes with energy computed: {len(ratios)}")

    names = [n for n in coords if n in ratios]
    xy = np.array([coords[n][:2] for n in names])   # top-down: x (L-R), y (post-ant)
    vals = np.array([ratios[n] for n in names])

    fig, ax = plt.subplots(figsize=(9, 10))

    # schematic brain outline (axial, top-down) - ellipse sized to the coordinates
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    rx = max(80, np.abs(xy[:, 0] - cx).max() * 1.4)
    ry = max(100, np.abs(xy[:, 1] - cy).max() * 1.4)
    brain = Ellipse((cx, cy), 2 * rx, 2 * ry, facecolor="#f2f2f2",
                    edgecolor="#888888", lw=1.5, zorder=0)
    ax.add_patch(brain)
    # midline + front marker
    ax.plot([cx, cx], [cy - ry, cy + ry], color="#cccccc", lw=1, zorder=1)
    ax.text(cx, cy + ry * 1.02, "ANTERIOR", ha="center", fontsize=9, color="#666")
    ax.text(cx, cy - ry * 1.06, "POSTERIOR", ha="center", fontsize=9, color="#666")
    ax.text(cx - rx * 1.05, cy, "L", ha="center", va="center", fontsize=11, color="#666")
    ax.text(cx + rx * 1.05, cy, "R", ha="center", va="center", fontsize=11, color="#666")

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="inferno",
                    s=140, edgecolor="black", lw=0.6, zorder=3)
    # label electrodes
    for n, (x, y) in zip(names, xy):
        ax.text(x, y + 2, n, fontsize=6, ha="center", va="bottom", zorder=4)

    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Ictal energy ratio (dB)  -  higher = more involved", fontsize=10)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Seizure involvement map (top-down) - {SUBJECT}\n"
                 f"warm = electrode strongly sees the seizure",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_PNG}")


if __name__ == "__main__":
    main()