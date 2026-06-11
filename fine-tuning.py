# fine_tuning.py — versione riscritta e robusta
#
# Cambiamenti principali rispetto alla versione precedente:
#   1. Parser onset/offset CORRETTO: "sz offset" non viene più scambiato per onset
#      (OFFSET controllato prima di ONSET; lookbehind per non matchare "off"+onset).
#   2. Accoppiamento onset/offset ROBUSTO: gestisce crisi multiple, offset mancanti,
#      offset orfani, durate implausibili. Niente più "primo offset successivo" a caso.
#   3. Ramo OMNI rimosso (non utilizzato) — eliminato il bug (0, 1e9) = tutto-ictale.
#   4. Preprocessing allineato a LUNA: bandpass 0.1–75 Hz + notch (configurabile),
#      z-score per canale (come nel pre-training), applicati nell'ordine corretto.
#   5. Selezione canali deterministica (no più hash() randomizzato) e senza
#      zero-padding distruttivo: se i canali sono pochi, si usano quelli che ci sono.
#   6. file_seed deterministico via hashlib (riproducibile tra esecuzioni).

import re
import hashlib
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from safetensors.torch import load_file
from braindecode.models import LUNA
from braindecode.util import set_random_seeds
from tqdm import tqdm
import random
import csv
import yaml
import wandb

# SWECDataset è opzionale: importalo solo se il file esiste
try:
    from swec_dataset import SWECDataset
    _HAS_SWEC_MODULE = True
except Exception:
    _HAS_SWEC_MODULE = False

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configurazione
# ──────────────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BIDS_ROOTS = [Path(p) for p in config["bids_roots"]]
SWEC_ROOT = Path(config.get("swec_root", ""))
SWEC_TEST_SUBJECTS = set(config.get("swec_test_subjects", []))
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
GRAD_ACCUM_STEPS = config.get("grad_accum_steps", 1)

# Preprocessing allineato a LUNA (con default sensati se non in config)
BANDPASS_LOW = config.get("bandpass_low", 0.1)
BANDPASS_HIGH = config.get("bandpass_high", 75.0)
NOTCH_FREQ = config.get("notch_freq", 50.0)   # 50 Hz Europa (SWEC/Berna), 60 Hz USA (HUP)
# Limiti clinici per validare gli intervalli di crisi (secondi)
MAX_SEIZURE_DUR = config.get("max_seizure_dur", 300.0)
DEFAULT_SEIZURE_DUR = config.get("default_seizure_dur", 120.0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seeds(SEED, cuda=torch.cuda.is_available())
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. wandb
# ──────────────────────────────────────────────────────────────────────────────
wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=config["wandb"].get("run_name"),
    tags=config["wandb"].get("tags", []),
    config=config,
)
wandb.config.update({
    "device": DEVICE,
    "n_times": N_TIMES,
    "n_chans_out": N_CHANS_OUT,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "window_sec": WINDOW_SEC,
    "bandpass": [BANDPASS_LOW, BANDPASS_HIGH],
    "notch_freq": NOTCH_FREQ,
    "swec_enabled": SWEC_ROOT.exists(),
})

# ──────────────────────────────────────────────────────────────────────────────
# 3. Discovery soggetti e split
# ──────────────────────────────────────────────────────────────────────────────
print("DATASET DISCOVERY")
ALL_SUBJECTS = {}
for root in BIDS_ROOTS:
    if not root.exists():
        print(f"{root} NOT FOUND, skipping")
        continue
    subjects = sorted([d.name for d in root.glob("sub-*") if d.is_dir()])
    ALL_SUBJECTS[root.name] = subjects
    print(f"{root.name}: {len(subjects)} subjects")

total_subjects = sum(len(s) for s in ALL_SUBJECTS.values())
print(f"\nTotal BIDS subjects: {total_subjects}\n")
print(f"Test subjects (held-out): {', '.join(TEST_SUBJECTS)}\n")

SWEC_ALL_IDS = []
if SWEC_ROOT.exists():
    SWEC_ALL_IDS = sorted([d.name for d in SWEC_ROOT.iterdir() if d.is_dir()])
    print(f"SWEC_ETHZ: {len(SWEC_ALL_IDS)} soggetti trovati")
    print(f"SWEC test subjects (held-out): {', '.join(SWEC_TEST_SUBJECTS)}\n")
else:
    print(f"⚠ SWEC root non trovata ({SWEC_ROOT}), verrà saltato.\n")

VAL_SUBJECTS, TRAIN_SUBJECTS = [], []
for ds_name, subs in ALL_SUBJECTS.items():
    eligible = [s for s in subs if s not in TEST_SUBJECTS]
    if not eligible:
        continue
    rng_shuffle = random.Random(SEED)
    eligible_shuffled = rng_shuffle.sample(eligible, len(eligible))
    n_val = max(1, int(len(eligible_shuffled) * 0.2))
    VAL_SUBJECTS.extend(eligible_shuffled[:n_val])
    TRAIN_SUBJECTS.extend(eligible_shuffled[n_val:])
    print(f"{ds_name}: eligible={len(eligible)}, val={n_val}, train={len(eligible_shuffled)-n_val}")

SWEC_TRAIN_IDS, SWEC_VAL_IDS = [], []
if SWEC_ALL_IDS:
    swec_eligible = [s for s in SWEC_ALL_IDS if s not in SWEC_TEST_SUBJECTS]
    rng_swec = random.Random(SEED)
    swec_shuffled = rng_swec.sample(swec_eligible, len(swec_eligible))
    n_val_swec = max(1, int(len(swec_shuffled) * 0.2))
    SWEC_VAL_IDS = swec_shuffled[:n_val_swec]
    SWEC_TRAIN_IDS = swec_shuffled[n_val_swec:]
    print(f"\nSWEC_ETHZ: eligible={len(swec_eligible)}, val={len(SWEC_VAL_IDS)}, train={len(SWEC_TRAIN_IDS)}")

wandb.config.update({
    "swec_train_subjects": len(SWEC_TRAIN_IDS),
    "swec_val_subjects": len(SWEC_VAL_IDS),
}, allow_val_change=True)

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
val_paths = get_subject_paths(VAL_SUBJECTS)
print(f"\nTraining:   {len(train_paths)} subjects")
print(f"Validation: {len(val_paths)} subjects")
print(f"Test:       {len(TEST_SUBJECTS)} subjects (hold-out)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Parsing eventi — CORRETTO E ROBUSTO
# ──────────────────────────────────────────────────────────────────────────────
# OFFSET deve essere controllato PRIMA di ONSET, altrimenti "sz offset" matcha \bsz\b.
# Il lookbehind (?<!off) evita che "onset" matchi dentro "offset".
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
    """Ritorna 'ONSET', 'OFFSET' o None. OFFSET ha priorità."""
    l = str(label).strip().lower()
    if OFFSET_RE.search(l):
        return "OFFSET"
    if ONSET_RE.search(l):
        return "ONSET"
    return None

def parse_seizure_intervals(events_tsv: Path,
                            max_dur: float = MAX_SEIZURE_DUR,
                            default_dur: float = DEFAULT_SEIZURE_DUR):
    """Estrae intervalli (t_start, t_stop) robusti da un events.tsv BIDS.

    Gestisce: crisi multiple, offset mancanti, offset orfani, durate implausibili.
    """
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
            # onset già aperto senza offset → chiudi col default (troncato al nuovo onset)
            if pending_onset is not None:
                intervals.append((pending_onset, min(pending_onset + default_dur, t)))
            pending_onset = t
        else:  # OFFSET
            if pending_onset is not None:
                dur = t - pending_onset
                if dur <= 0 or dur > max_dur:
                    intervals.append((pending_onset, pending_onset + default_dur))
                else:
                    intervals.append((pending_onset, t))
                pending_onset = None
            # offset orfano (nessun onset aperto) → ignorato
    if pending_onset is not None:
        intervals.append((pending_onset, pending_onset + default_dur))
    return intervals

# ──────────────────────────────────────────────────────────────────────────────
# 5. Caricamento e PREPROCESSING allineato a LUNA
# ──────────────────────────────────────────────────────────────────────────────
def load_run(file_path: Path):
    if file_path.suffix == ".edf":
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
    elif file_path.suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(file_path), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")

    # Seleziona canali 'good' dal channels.tsv se presente
    ch_file = file_path.parent / file_path.name.replace(
        "_ieeg.edf", "_channels.tsv").replace("_ieeg.vhdr", "_channels.tsv")
    if ch_file.exists():
        ch_df = pd.read_csv(ch_file, sep="\t")
        if "status" in ch_df.columns:
            good = ch_df[ch_df["status"] == "good"]["name"].tolist()
            available = [c for c in good if c in raw.ch_names]
            if available:
                raw.pick(available)

    # --- Preprocessing nell'ordine corretto (come LUNA) ---
    sfreq_now = raw.info["sfreq"]
    n_samples = raw.n_times
    # Un filtro FIR a 0.1 Hz richiede molti campioni: se il segnale è troppo
    # corto, MNE costruisce un filtro più lungo del segnale -> distorsione.
    # Stima grezza della lunghezza filtro per l_freq: ~3.3 * sfreq / l_freq.
    min_len_for_filter = int(3.3 * sfreq_now / BANDPASS_LOW)
    can_filter = n_samples > min_len_for_filter

    # 1) Notch sul rumore di rete (solo se il segnale è abbastanza lungo)
    if n_samples > int(3.3 * sfreq_now / 1.0):  # notch ha banda ~1 Hz
        try:
            raw.notch_filter(freqs=[NOTCH_FREQ], verbose=False)
        except Exception as e:
            print(f"  ⚠ notch_filter saltato su {file_path.name}: {e}")
    # 2) Bandpass 0.1–75 Hz (solo se il segnale è abbastanza lungo)
    if can_filter:
        nyq = sfreq_now / 2.0
        h_freq = min(BANDPASS_HIGH, nyq - 1.0)
        raw.filter(l_freq=BANDPASS_LOW, h_freq=h_freq, verbose=False)
    else:
        print(f"  ⚠ Filtraggio saltato su {file_path.name}: "
              f"segnale troppo corto ({n_samples} campioni)")
    # 3) Resampling a 256 Hz (anti-alias gestito da MNE). Dopo il filtraggio.
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 1:
        raw.resample(TARGET_SFREQ, verbose=False)
    return raw

def stable_seed(path: Path) -> int:
    """Seed deterministico tra esecuzioni (hash() è randomizzato; hashlib no)."""
    h = hashlib.md5(str(path.absolute()).encode()).hexdigest()
    return (SEED + int(h, 16)) % (2**32)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Windowing — z-score per canale, gestione canali senza zero-padding
# ──────────────────────────────────────────────────────────────────────────────
def extract_windows(data: np.ndarray, sfreq: int, ictal_intervals: list, file_seed: int):
    n_chans, total = data.shape
    step = int(N_TIMES * (1 - OVERLAP))
    rng_file = np.random.default_rng(file_seed)

    # Selezione canali decisa UNA volta per file (non per finestra), deterministica.
    # LUNA è istanziato con n_chans=N_CHANS_OUT fisso, e il collate del DataLoader
    # richiede che tutte le finestre abbiano la STESSA forma. Quindi produciamo
    # sempre esattamente N_CHANS_OUT canali:
    #   - se ne abbiamo di più  -> sottocampioniamo senza rimpiazzo
    #   - se ne abbiamo di meno -> RIPETIAMO ciclicamente i canali reali
    #     (meglio dello zero-padding: niente segnale piatto artificiale che
    #      inquina la cross-attention; diamo solo informazione vera, ridondante)
    if n_chans >= N_CHANS_OUT:
        chan_idx = np.sort(rng_file.choice(n_chans, N_CHANS_OUT, replace=False))
    else:
        reps = int(np.ceil(N_CHANS_OUT / n_chans))
        chan_idx = np.tile(np.arange(n_chans), reps)[:N_CHANS_OUT]

    windows = []
    for start in range(0, total - N_TIMES + 1, step):
        end = start + N_TIMES
        t_start, t_end = start / sfreq, end / sfreq

        label = 0
        for (on, off) in ictal_intervals:
            overlap_s = max(0, min(t_end, off) - max(t_start, on))
            if overlap_s / (t_end - t_start) > 0.5:
                label = 1
                break

        win = data[chan_idx, start:end]

        # z-score per canale (come nel pre-training di LUNA), robusto a std nulla
        mean = win.mean(axis=1, keepdims=True)
        std = win.std(axis=1, keepdims=True) + 1e-6
        win = (win - mean) / std

        windows.append((win.astype(np.float32), label))
    return windows

# ──────────────────────────────────────────────────────────────────────────────
# 7. Dataset
# ──────────────────────────────────────────────────────────────────────────────
class IctalDataset(Dataset):
    def __init__(self, subject_paths, dataset_name=""):
        self.samples = []
        print(f"Loading {dataset_name} dataset...")
        for sub_path in tqdm(subject_paths, desc=f"Processing {dataset_name} subjects"):
            # Sessione: prova ses-presurgery, poi ieeg diretto
            ieeg_dir = sub_path / "ses-presurgery" / "ieeg"
            if not ieeg_dir.exists():
                ieeg_dir = sub_path / "ieeg"
                if not ieeg_dir.exists():
                    continue

            # Cerca BrainVision, poi EDF
            files = sorted(ieeg_dir.glob("*_ieeg.vhdr")) or sorted(ieeg_dir.glob("*_ieeg.edf"))
            for ieeg_file in files:
                ev_file = Path(re.sub(r"_ieeg\.(vhdr|edf)$", "_events.tsv", str(ieeg_file)))
                ictal_intervals = parse_seizure_intervals(ev_file)
                try:
                    raw = load_run(ieeg_file)
                except Exception as e:
                    print(f"Skip {sub_path.name}/{ieeg_file.name}: {e}")
                    continue
                wins = extract_windows(raw.get_data(), TARGET_SFREQ,
                                       ictal_intervals, stable_seed(ieeg_file))
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

# ──────────────────────────────────────────────────────────────────────────────
# 8. Caricamento LUNA (con adattamento posizionale se N_TIMES cambia)
# ──────────────────────────────────────────────────────────────────────────────
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
    model = LUNA(n_outputs=2, n_chans=N_CHANS_OUT, n_times=N_TIMES,
                 embed_dim=64, num_queries=4, depth=8)
    raw_sd = load_file(str(CKPT_PATH), device="cpu")
    sd = {}
    old_len = 2000
    for k, v in raw_sd.items():
        if k in SKIP_KEYS:
            continue
        new_k = KEY_MAP.get(k, k)
        if v.dim() >= 2 and v.shape[-1] == old_len and old_len != N_TIMES:
            v_interp = torch.nn.functional.interpolate(
                v.unsqueeze(0) if v.dim() == 2 else v,
                size=N_TIMES, mode="linear", align_corners=False
            ).squeeze(0)
            sd[new_k] = v_interp
            print(f"  ⚠ Interpolated {k}: {old_len} → {N_TIMES}")
        else:
            sd[new_k] = v
    model.load_state_dict(sd, strict=False)
    print(f"✓ Pre-trained weights loaded (adapted for n_times={N_TIMES})")
    return model

model = load_luna().to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# 9. Datasets
# ──────────────────────────────────────────────────────────────────────────────
print("LOADING TRAINING DATASET")
bids_train_ds = IctalDataset(train_paths, dataset_name="BIDS TRAIN")
print("\nLOADING VALIDATION DATASET")
bids_val_ds = IctalDataset(val_paths, dataset_name="BIDS VAL")

if SWEC_ROOT.exists() and SWEC_TRAIN_IDS and _HAS_SWEC_MODULE:
    print("\nLOADING SWEC TRAINING DATASET")
    swec_train_ds = SWECDataset(SWEC_ROOT, SWEC_TRAIN_IDS, dataset_name="SWEC TRAIN")
    print("\nLOADING SWEC VALIDATION DATASET")
    swec_val_ds = SWECDataset(SWEC_ROOT, SWEC_VAL_IDS, dataset_name="SWEC VAL")
    train_ds = ConcatDataset([bids_train_ds, swec_train_ds])
    val_ds = ConcatDataset([bids_val_ds, swec_val_ds])
    print(f"\nDataset combinati → Train: {len(train_ds)}, Val: {len(val_ds)} finestre")
else:
    train_ds = bids_train_ds
    val_ds = bids_val_ds
    if not _HAS_SWEC_MODULE:
        print("\n⚠ Modulo swec_dataset non importabile, training solo su BIDS")
    else:
        print("\n⚠ SWEC non caricato, training solo su BIDS")

def get_labels(ds):
    if isinstance(ds, ConcatDataset):
        labels = []
        for sub_ds in ds.datasets:
            labels.extend(get_labels(sub_ds))
        return labels
    return [ds[i][1].item() for i in range(len(ds))]

labels = get_labels(train_ds)
n_ictal = sum(labels)
n_inter = len(labels) - n_ictal
n_total = max(n_ictal + n_inter, 1)
class_weight_0 = n_total / (2 * n_inter) if n_inter > 0 else 1.0
class_weight_1 = n_total / (2 * n_ictal) if n_ictal > 0 else 1.0
print(f"Class distribution - Interictal: {n_inter}, Ictal: {n_ictal}")
print(f"Class weights - class 0: {class_weight_0:.3f}, class 1: {class_weight_1:.3f}\n")

if n_ictal == 0:
    print("⚠⚠ ATTENZIONE: ZERO finestre ictali nel training set. "
          "Controlla i parser/eventi: il modello non può imparare nulla.\n")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────────────────────────────────────
# 10. Loss e optimizer
# ──────────────────────────────────────────────────────────────────────────────
class_weights = torch.tensor([class_weight_0, class_weight_1], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

head_params = list(model.final_layer.parameters())
backbone_params = [p for n, p in model.named_parameters() if not n.startswith("final_layer")]

def get_optimizer(freeze_backbone: bool):
    for p in backbone_params:
        p.requires_grad = not freeze_backbone
    if freeze_backbone:
        return torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=1e-4)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=1e-4)

optimizer = get_optimizer(freeze_backbone=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

metrics_file = SAVE_DIR / "training_metrics.csv"
metrics_file.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_file, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "train_f1", "val_prec", "val_rec", "val_f1"])

# ──────────────────────────────────────────────────────────────────────────────
# 11. Training loop (gradient accumulation + freeze/unfreeze backbone)
# ──────────────────────────────────────────────────────────────────────────────
best_val_f1 = 0.0
patience_counter = 0

print("STARTING TRAINING")
for epoch in range(1, N_EPOCHS + 1):
    if epoch == FROZEN_EPOCHS + 1:
        print(f"\n>>> Epoch {epoch}: unfreezing backbone")
        optimizer = get_optimizer(freeze_backbone=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS - FROZEN_EPOCHS)

    model.train()
    total_loss, tp, fp, fn = 0.0, 0, 0, 0
    optimizer.zero_grad()

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
    for step, (X, y) in enumerate(train_loop):
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y) / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=1)
        total_loss += (loss.item() * GRAD_ACCUM_STEPS) * len(y)
        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()
        train_loop.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

    avg_loss = total_loss / max(len(train_ds), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    train_f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    scheduler.step()

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

    wandb.log({
        "epoch": epoch,
        "train/loss": avg_loss,
        "train/precision": prec,
        "train/recall": rec,
        "train/f1": train_f1,
        "val/precision": v_prec,
        "val/recall": v_rec,
        "val/f1": val_f1,
        "lr_backbone": 0 if epoch <= FROZEN_EPOCHS else LR_BACKBONE,
    })

    with open(metrics_file, "a", newline="") as f:
        csv.writer(f).writerow([epoch, avg_loss, train_f1, v_prec, v_rec, val_f1])

    print(f"Epoch {epoch:2d} | Tr.loss={avg_loss:.4f} F1={train_f1:.3f} | "
          f"Val prec={v_prec:.3f} rec={v_rec:.3f} F1={val_f1:.3f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), SAVE_DIR / "best.pt")
        print(f"  ✓ Saved (val F1={val_f1:.4f})")
        if config["wandb"].get("log_model", True):
            artifact = wandb.Artifact(name="luna_ieeg_best", type="model",
                                      metadata={"val_f1": val_f1, "epoch": epoch})
            artifact.add_file(str(SAVE_DIR / "best.pt"))
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