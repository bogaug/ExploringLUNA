#!/usr/bin/env python3
"""
Ispeziona gli events.tsv di ogni dataset BIDS e conta i marker di crisi.
Lancialo dalla cartella del progetto (dove c'e' ./dataset).
Dice, per ogni dataset: quanti events.tsv, quanti contengono crisi,
e quali trial_type unici compaiono (per capire lo schema di annotazione).
"""
import re
from pathlib import Path
from collections import Counter
import pandas as pd

DATASET_ROOT = Path("./dataset")
BIDS_DATASETS = ["ds003029", "ds003844", "ds004100", "ds005398", "ds007095", "ds005448"]

# Stesse regex del fine-tuning (versione corretta: OFFSET prima di ONSET)
OFFSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*(?:offset|end)"
    r"|\boffset\b|seizure\s*off|devolution|\bend\b", re.I)
ONSET_RE = re.compile(
    r"(?:sz|seizure|ictal|clinical|electrographic)\s*onset"
    r"|(?<!off)\bonset\b|^sz$|sz\s*event", re.I)

def classify(label):
    l = str(label).strip().lower()
    if OFFSET_RE.search(l): return "OFFSET"
    if ONSET_RE.search(l):  return "ONSET"
    return None

print(f"{'DATASET':<12} {'#tsv':>6} {'#con_crisi':>11} {'#onset':>8} {'#offset':>8}   trial_type piu' comuni")
print("-" * 100)

for ds in BIDS_DATASETS:
    root = DATASET_ROOT / ds
    if not root.exists():
        print(f"{ds:<12}  -- cartella non trovata --")
        continue
    tsv_files = list(root.rglob("*_events.tsv"))
    n_tsv = len(tsv_files)
    n_with_sz = 0
    n_onset = n_offset = 0
    trial_types = Counter()
    for tsv in tsv_files:
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "trial_type" not in df.columns:
            continue
        file_has_sz = False
        for v in df["trial_type"].astype(str):
            trial_types[v.strip().lower()] += 1
            k = classify(v)
            if k == "ONSET":
                n_onset += 1; file_has_sz = True
            elif k == "OFFSET":
                n_offset += 1; file_has_sz = True
        if file_has_sz:
            n_with_sz += 1
    top = ", ".join(f"{t}({c})" for t, c in trial_types.most_common(4))
    print(f"{ds:<12} {n_tsv:>6} {n_with_sz:>11} {n_onset:>8} {n_offset:>8}   {top}")

print("\nLEGENDA: i dataset con #con_crisi = 0 NON hanno crisi annotate")
print("         -> vanno TOLTI da bids_roots nel config (diluiscono il training).")