"""
glass_brain.py
Three orthogonal projections of intracranial electrodes on the fsaverage brain,
colored by ictal energy (how strongly each electrode "sees" the seizure, dB).

Instead of relying on a 3D camera, the electrode 3D coordinates are projected
directly onto the three anatomical planes:
  - Axial    (top-down):  x (L-R)  vs y (post-ant)
  - Sagittal (lateral):   y (post-ant) vs z (inf-sup)
  - Coronal  (posterior): x (L-R)  vs z (inf-sup)
The fsaverage pial surface vertices are projected onto the same plane to draw
a faint brain silhouette as background. Depth (SEEG) electrodes that overlap in
one plane are separated in the others. Deterministic, no 3D rendering needed.

Usage:
    python glass_brain.py
"""

import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
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
OUT_PNG = "glass_brain.png"
TOP_LABELS = 6


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


# ------------------------------ Electrode coords (fsaverage, mm) ------------------------------
def load_electrodes_mm(subject):
    ieeg_dir = BIDS_ROOT / subject / "ses-presurgery" / "ieeg"
    elec_files = list(ieeg_dir.glob("*_electrodes.tsv"))
    if not elec_files:
        raise FileNotFoundError("No *_electrodes.tsv found")
    df = pd.read_csv(elec_files[0], sep="\t")
    coords = {}
    for _, row in df.iterrows():
        try:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        except (ValueError, TypeError):
            continue
        if np.isfinite([x, y, z]).all():
            coords[str(row["name"])] = np.array([x, y, z])
    # ensure millimeters (fsaverage surf is in mm); convert if values look like meters
    mags = np.array([np.abs(v).max() for v in coords.values()])
    if mags.size and np.median(mags) < 1.0:
        coords = {k: v * 1000.0 for k, v in coords.items()}
    return coords


# ------------------------------ fsaverage pial surface vertices (mm) ------------------------------
def load_brain_vertices():
    subjects_dir = Path(os.path.dirname(mne.datasets.fetch_fsaverage(verbose=False)))
    surf_dir = subjects_dir / "fsaverage" / "surf"
    verts = []
    for hemi in ["lh.pial", "rh.pial"]:
        fpath = surf_dir / hemi
        if fpath.exists():
            v, _ = mne.read_surface(str(fpath))   # vertices in surface RAS (mm)
            verts.append(v)
    if not verts:
        return None
    return np.vstack(verts)


# ------------------------------ Ictal energy per electrode (dB) ------------------------------
def ictal_energy_db(raw, intervals, coords):
    data = raw.get_data(); ch_names = raw.ch_names; sf = raw.info["sfreq"]
    total = data.shape[1]
    ictal = np.zeros(total, dtype=bool)
    for on, off in intervals:
        ictal[max(0, int(on * sf)):min(total, int(off * sf))] = True
    if ictal.sum() == 0 or (~ictal).sum() == 0:
        return {}
    out = {}
    for name in coords:
        if name not in ch_names:
            continue
        sig = data[ch_names.index(name)]
        p_ict = np.mean(sig[ictal] ** 2)
        p_bkg = np.mean(sig[~ictal] ** 2) + 1e-20
        out[name] = 10 * np.log10((p_ict / p_bkg) + 1e-12)
    return out


# ------------------------------ Main ------------------------------
def main():
    print(f"Subject: {SUBJECT}")
    coords = load_electrodes_mm(SUBJECT)
    print(f"Electrodes with coordinates: {len(coords)}")

    brain = load_brain_vertices()
    if brain is not None:
        print(f"Brain silhouette vertices: {len(brain)}")
        # subsample for speed
        if len(brain) > 20000:
            idx = np.random.default_rng(0).choice(len(brain), 20000, replace=False)
            brain = brain[idx]

    ieeg_dir = BIDS_ROOT / SUBJECT / "ses-presurgery" / "ieeg"
    files = sorted(ieeg_dir.glob("*_ieeg.vhdr")) or sorted(ieeg_dir.glob("*_ieeg.edf"))
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
    energy = ictal_energy_db(raw, intervals, coords)
    names = [n for n in coords if n in energy]
    if not names:
        raise RuntimeError("No electrode names match between coords and signal")
    pos = np.array([coords[n] for n in names])   # (N,3) in mm
    vals = np.array([energy[n] for n in names])
    print(f"Electrodes plotted: {len(names)}")

    vmin, vmax = float(vals.min()), float(vals.max())

    # three orthogonal projections: (axis_x, axis_y, title, xlabel, ylabel)
    planes = [
        (0, 1, "Axial (top-down)", "L  <->  R", "post <->  ant"),
        (1, 2, "Sagittal (lateral)", "post <->  ant", "inf <->  sup"),
        (0, 2, "Coronal (posterior)", "L  <->  R", "inf <->  sup"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    sc = None
    for ax, (ix, iy, title, xlab, ylab) in zip(axes, planes):
        if brain is not None:
            ax.scatter(brain[:, ix], brain[:, iy], s=1, c="#340202",
                       alpha=0.10, edgecolors="none", zorder=0)
        sc = ax.scatter(pos[:, ix], pos[:, iy], c=vals, cmap="inferno",
                        s=90, edgecolor="black", lw=0.5, zorder=5,
                        vmin=vmin, vmax=vmax)
        # label the most-involved electrodes
        for i in np.argsort(vals)[-TOP_LABELS:]:
            ax.text(pos[i, ix], pos[i, iy] + 3, names[i], fontsize=7,
                    ha="center", zorder=6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Ictal energy ratio (dB) - higher = more involved", fontsize=10)
    fig.suptitle(f"Electrode seizure involvement - {SUBJECT}",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_PNG}")


if __name__ == "__main__":
    main()