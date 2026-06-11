#!/usr/bin/env python3
"""
Verifica integrità dataset iEEG (BIDS e SWEC-ETHZ) per ExploringLUNA.

Robusto a priori: non assume nomi di colonna, layout di sessione, o
frequenze di campionamento specifiche. Distingue "nessuna crisi annotata"
(neutro, es. dataset di stimolazione) da problemi strutturali reali.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import h5py
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────
REQUIRED_BIDS_ROOT_FILES = ["dataset_description.json", "participants.tsv"]
REQUIRED_EVENTS_COLUMNS  = {"onset", "duration"}
VALID_IEEG_EXTENSIONS    = {".edf", ".vhdr", ".set", ".fif", ".nwb", ".mefd", ".bdf", ".cnt"}

# Colonne testuali in cui può comparire un'etichetta di crisi (qualsiasi BIDS)
LABEL_COLUMNS = ["trial_type", "value", "eventtype", "event_type",
                 "description", "label", "type", "annotation", "note", "notes"]

# Keyword crisi con word-boundary dove serve (evita match dentro altre parole)
SEIZURE_PATTERNS = [
    r"\bseizure\b", r"\bictal\b", r"\bsz\b", r"\bseiz\b",
    r"\bepilep", r"\bonset\b", r"\boffset\b", r"\beeg onset\b",
    r"\bclinical onset\b", r"\belectrographic\b", r"\bsz event\b",
]
SEIZURE_RE = re.compile("|".join(SEIZURE_PATTERNS), re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ok(msg):   return f"{Fore.GREEN}\u2714{Style.RESET_ALL}  {msg}"
def warn(msg): return f"{Fore.YELLOW}\u26a0{Style.RESET_ALL}  {msg}"
def err(msg):  return f"{Fore.RED}\u2718{Style.RESET_ALL}  {msg}"
def info(msg): return f"{Fore.CYAN}\u2139{Style.RESET_ALL}  {msg}"
_LINE = "\u2500" * 60
_NL = "\n"
def hdr(msg):  return f"{_NL}{Fore.WHITE}{Style.BRIGHT}{_LINE}{_NL}  {msg}{_NL}{_LINE}{Style.RESET_ALL}"


def find_seizure_events(df: pd.DataFrame):
    """
    Cerca crisi in TUTTE le colonne testuali plausibili, non solo trial_type.
    Ritorna (n_seizure_events, colonna_usata, sample_labels).
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    best_n, best_col, best_labels = 0, None, []
    for key in LABEL_COLUMNS:
        if key in cols_lower:
            col = cols_lower[key]
            vals = df[col].dropna().astype(str)
            if vals.empty:
                continue
            # mask booleana esplicita: evita .sum() su Series non-numeriche/vuote
            mask = vals.map(lambda x: bool(SEIZURE_RE.search(x))).astype(bool)
            n = int(mask.to_numpy().sum())
            if n > best_n:
                best_n, best_col = n, col
                best_labels = vals[mask].unique().tolist()[:10]
    return best_n, best_col, best_labels


# ─────────────────────────────────────────────────────────────
# CONTROLLO SINGOLO events.tsv
# ─────────────────────────────────────────────────────────────
def check_events_file(tsv_path: Path) -> dict:
    result = {"path": str(tsv_path), "ok": True, "issues": [], "stats": {}}
    try:
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"Parsing fallito: {e}")
        return result

    df.columns = [c.strip() for c in df.columns]
    cols = set(df.columns)

    missing = REQUIRED_EVENTS_COLUMNS - {c.lower() for c in cols}
    if missing:
        result["ok"] = False
        result["issues"].append(f"Colonne obbligatorie mancanti: {missing}")

    # onset numerico
    onset_col = next((c for c in cols if c.lower() == "onset"), None)
    if onset_col:
        non_numeric = pd.to_numeric(df[onset_col], errors="coerce").isna().sum()
        if non_numeric > 0:
            result["issues"].append(f"onset: {non_numeric} valori non numerici")

    # ricerca crisi su tutte le colonne testuali
    n_sz, sz_col, sz_labels = find_seizure_events(df)
    result["stats"]["n_rows"]            = len(df)
    result["stats"]["columns"]           = list(cols)
    result["stats"]["n_seizure_events"]  = n_sz
    result["stats"]["seizure_column"]    = sz_col
    result["stats"]["seizure_labeled"]   = n_sz > 0
    result["stats"]["seizure_label_sample"] = sz_labels
    return result


# ─────────────────────────────────────────────────────────────
# CONTROLLO DATASET BIDS
# ─────────────────────────────────────────────────────────────
def check_bids_dataset(ds_path: Path, verbose: bool) -> dict:
    report = {"name": ds_path.name, "type": "BIDS", "path": str(ds_path),
              "score": 0, "issues": [], "warnings": [], "events": [], "stats": {}}

    # Root files
    for f in REQUIRED_BIDS_ROOT_FILES:
        fp = ds_path / f
        if not fp.exists():
            report["warnings"].append(f"File root assente: {f}")
        elif f == "dataset_description.json":
            try:
                desc = json.loads(fp.read_text())
                report["stats"]["bids_name"] = desc.get("Name", "?")
                report["stats"]["bids_ver"]  = desc.get("BIDSVersion", "?")
            except Exception as e:
                report["warnings"].append(f"dataset_description.json non parsabile: {e}")
        elif f == "participants.tsv":
            try:
                report["stats"]["n_subjects"] = len(pd.read_csv(fp, sep="\t"))
            except Exception:
                pass

    # Soggetti
    subject_dirs = sorted(d for d in ds_path.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))
    report["stats"]["n_subject_dirs"] = len(subject_dirs)
    if not subject_dirs:
        report["issues"].append("Nessuna cartella sub-* trovata")

    # Scansione: supporta ses-* opzionale e ieeg/ o eeg/
    ieeg_files, events_files = [], []
    for sub in subject_dirs:
        ses_dirs = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith("ses-")]
        for base in (ses_dirs if ses_dirs else [sub]):
            for modality in ("ieeg", "eeg"):
                mdir = base / modality
                if not mdir.exists():
                    continue
                for f in mdir.iterdir():
                    if f.suffix in VALID_IEEG_EXTENSIONS:
                        ieeg_files.append(f)
                    if f.name.endswith("_events.tsv"):
                        events_files.append(f)

    report["stats"]["n_ieeg_files"]   = len(ieeg_files)
    report["stats"]["n_events_files"] = len(events_files)

    # Analisi eventi
    total_sz = 0
    n_with_sz = 0
    for ev_path in events_files:
        ev = check_events_file(ev_path)
        report["events"].append(ev)
        total_sz += ev["stats"].get("n_seizure_events", 0)
        if ev["stats"].get("seizure_labeled"):
            n_with_sz += 1

    report["stats"]["total_seizure_events"] = total_sz
    report["stats"]["n_events_with_seizure"] = n_with_sz
    report["stats"]["has_seizure_annotations"] = total_sz > 0

    if not ieeg_files:
        report["issues"].append("Nessun file iEEG riconosciuto")
    if not events_files:
        report["warnings"].append("Nessun *_events.tsv trovato")
    elif total_sz == 0:
        # NEUTRO: non un errore. Es. dataset di stimolazione (SPES).
        report["warnings"].append(
            "Nessuna crisi annotata trovata (potrebbe essere normale, "
            "es. dataset di stimolazione) \u2014 non usabile come test ictale")

    # Score: penalizza solo problemi STRUTTURALI, non l'assenza di crisi
    score = 100
    score -= len(report["issues"]) * 20
    score -= len(report["warnings"]) * 4
    report["score"] = max(0, min(100, score))
    return report


# ─────────────────────────────────────────────────────────────
# CONTROLLO SWEC-ETHZ (.h5)  — struttura reale: data/ieeg, data/seizures
# ─────────────────────────────────────────────────────────────
def _h5_get(f, path):
    """Accesso sicuro a un path annidato (es. 'data/ieeg'). None se assente."""
    try:
        return f[path]
    except (KeyError, ValueError):
        return None


def check_swec_dataset(ds_path: Path, verbose: bool) -> dict:
    report = {"name": ds_path.name, "type": "SWEC-ETHZ (HDF5)", "path": str(ds_path),
              "score": 0, "issues": [], "warnings": [], "stats": {}}

    # Raggruppa per soggetto (cartelle ID*) — valuta i _total separatamente
    # Escludi cartelle nascoste/di servizio (es. .cache, .datalad)
    subject_dirs = sorted(d for d in ds_path.iterdir()
                          if d.is_dir() and not d.name.startswith("."))
    h5_all = sorted(p for p in ds_path.rglob("*.h5")
                    if not any(part.startswith(".") for part in p.parts))
    report["stats"]["n_h5_files"]   = len(h5_all)
    report["stats"]["n_subjects"]   = len(subject_dirs)
    if not h5_all:
        report["issues"].append("Nessun file .h5 trovato")
        report["score"] = 0
        return report

    subj_ok, subj_issues, subj_no_total = [], [], []
    total_seizures = 0
    srates = set()

    for sub in subject_dirs:
        total_files = sorted(sub.glob("*_total.h5"))
        part_files  = sorted(sub.glob("*_part_*.h5"))
        srec = {"subject": sub.name, "issues": []}

        if not total_files:
            # Stato legittimo: ci sono i part ma non il _total con le annotazioni.
            # Non è corruzione -> avviso, non problema critico.
            subj_no_total.append(sub.name)
            continue

        tot = total_files[0]
        try:
            with h5py.File(tot, "r") as f:
                # Segnali: data/ieeg (annidato)
                ieeg = _h5_get(f, "data/ieeg")
                if ieeg is None or not isinstance(ieeg, h5py.Dataset):
                    srec["issues"].append("data/ieeg assente o non è un dataset")
                else:
                    shape = ieeg.shape
                    srec["shape"] = list(shape)
                    if len(shape) != 2:
                        srec["issues"].append(f"data/ieeg shape {shape} \u2260 2D")

                # Sampling rate: ATTRIBUTO, non chiave (qualsiasi valore valido)
                sr = f.attrs.get("sampling_rate", None)
                if sr is None:
                    srec["issues"].append("attributo sampling_rate assente")
                else:
                    sr = int(sr)
                    srec["sfreq"] = sr
                    srates.add(sr)

                # Canali
                ch = f.attrs.get("channels", None)
                if ch is not None:
                    srec["channels"] = int(ch)

                # Crisi: data/seizures con campi onsets/offsets
                sz = _h5_get(f, "data/seizures")
                if sz is None or not isinstance(sz, h5py.Dataset):
                    srec["issues"].append("data/seizures assente in _total.h5")
                else:
                    arr = sz[:]
                    names = arr.dtype.names or ()
                    if "onsets" in names and "offsets" in names:
                        n = arr["onsets"].ravel().shape[0]
                        srec["n_seizures"] = int(n)
                        total_seizures += int(n)
                    else:
                        # fallback: conta righe
                        n = arr.shape[0]
                        srec["n_seizures"] = int(n)
                        total_seizures += int(n)
                        srec["issues"].append(
                            f"data/seizures senza campi onsets/offsets (dtype={arr.dtype})")
        except OSError as e:
            srec["issues"].append(f"File non apribile: {e}")

        srec["n_part_files"] = len(part_files)
        if srec["issues"]:
            subj_issues.append(srec)
        else:
            subj_ok.append(srec)

    report["stats"]["subjects_ok"]          = len(subj_ok)
    report["stats"]["subjects_with_issues"] = len(subj_issues)
    report["stats"]["subjects_no_total"]    = len(subj_no_total)
    report["stats"]["total_seizures"]       = total_seizures
    report["stats"]["sampling_rates_found"] = sorted(srates)
    report["stats"]["has_seizure_annotations"] = total_seizures > 0

    for s in subj_issues:
        for iss in s["issues"]:
            report["issues"].append(f"[{s['subject']}] {iss}")

    if subj_no_total:
        report["warnings"].append(
            f"Soggetti senza _total.h5 (solo part, niente annotazioni): "
            f"{', '.join(subj_no_total)}")

    if verbose:
        for s in subj_ok:
            print(ok(f"  {s['subject']}: {s.get('n_seizures','?')} crisi, "
                     f"{s.get('channels','?')} canali @ {s.get('sfreq','?')}Hz, "
                     f"{s['n_part_files']} part"))

    if total_seizures == 0 and not report["issues"]:
        report["warnings"].append("Nessuna crisi trovata in alcun _total.h5")

    score = 100
    score -= len(report["issues"]) * 12
    score -= len(subj_issues) * 4
    report["score"] = max(0, min(100, score))
    return report


# ─────────────────────────────────────────────────────────────
# RILEVAMENTO TIPO
# ─────────────────────────────────────────────────────────────
def detect_dataset_type(ds_path: Path) -> str:
    if (ds_path / "dataset_description.json").exists():
        return "bids"
    if any(ds_path.rglob("*.h5")):
        return "swec"
    if any(d.is_dir() and d.name.startswith("sub-") for d in ds_path.iterdir()):
        return "bids"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# STAMPA REPORT
# ─────────────────────────────────────────────────────────────
def print_dataset_report(r: dict, verbose: bool):
    score = r["score"]
    col = Fore.GREEN if score >= 80 else Fore.YELLOW if score >= 50 else Fore.RED
    print(hdr(f"{r['name']}  [{r['type']}]  Score: {col}{score}/100{Style.RESET_ALL}"))

    stat_rows = []
    for k, v in r.get("stats", {}).items():
        if isinstance(v, list) and len(v) > 6:
            v = str(v[:6])[:-1] + ", ...]"
        stat_rows.append([k, v])
    if stat_rows:
        print(tabulate(stat_rows, headers=["Statistica", "Valore"],
                       tablefmt="simple", maxcolwidths=[40, 60]))

    if r["issues"]:
        print(f"\n  {Fore.RED}Problemi critici:{Style.RESET_ALL}")
        for iss in r["issues"][:20]:
            print(f"  {err(iss)}")
    else:
        print(f"\n  {ok('Nessun problema strutturale')}")

    if r.get("warnings"):
        print(f"\n  {Fore.YELLOW}Avvisi:{Style.RESET_ALL}")
        for w in r["warnings"]:
            print(f"  {warn(w)}")

    events = r.get("events", [])
    if events and verbose:
        ev_table = []
        for ev in events:
            s = ev["stats"]
            ev_table.append([Path(ev["path"]).name, s.get("n_rows", "?"),
                             s.get("seizure_column") or "\u2014",
                             s.get("n_seizure_events", 0),
                             "; ".join(ev["issues"]) if ev["issues"] else "\u2014"])
        print(f"\n  {Fore.CYAN}events.tsv:{Style.RESET_ALL}")
        print(tabulate(ev_table,
                       headers=["File", "Righe", "Col. crisi", "N crisi", "Issues"],
                       tablefmt="simple", maxcolwidths=[40, 8, 16, 8, 45]))
    elif events:
        nsz = r["stats"].get("n_events_with_seizure", 0)
        tot_sz = r["stats"].get("total_seizure_events", 0)
        msg = f"{nsz}/{len(events)} events.tsv con crisi annotate ({tot_sz} crisi totali)"
        print(f"\n  {info(msg)}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Verifica integrità dataset iEEG (BIDS + SWEC)")
    p.add_argument("--datasets-dir", "-d", default="dataset",
                   help="Cartella dei dataset (default: ./dataset)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--json-out", "-o", default=None)
    p.add_argument("--only", default=None)
    args = p.parse_args()

    root = Path(args.datasets_dir)
    if not root.exists():
        print(err(f"Cartella non trovata: {root.resolve()}")); sys.exit(1)

    dataset_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and not d.name.startswith("."))
    if args.only:
        dataset_dirs = [d for d in dataset_dirs if d.name == args.only]
        if not dataset_dirs:
            print(err(f"Dataset '{args.only}' non trovato")); sys.exit(1)

    print(f"\n{Fore.WHITE}{Style.BRIGHT}ExploringLUNA \u2014 Dataset Integrity Check{Style.RESET_ALL}")
    print(info(f"Root: {root.resolve()}"))
    print(info(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}"))
    print(info(f"Dataset trovati: {len(dataset_dirs)}"))

    reports = []
    for ds in dataset_dirs:
        dtype = detect_dataset_type(ds)
        if dtype == "bids":
            rep = check_bids_dataset(ds, args.verbose)
        elif dtype == "swec":
            rep = check_swec_dataset(ds, args.verbose)
        else:
            rep = {"name": ds.name, "type": "unknown", "path": str(ds), "score": 0,
                   "issues": ["Tipo non riconosciuto"], "warnings": [], "stats": {}}
        print_dataset_report(rep, args.verbose)
        reports.append(rep)

    # Riepilogo
    print(hdr("RIEPILOGO"))
    rows = []
    for r in reports:
        s = r["score"]
        light = "\U0001f7e2" if s >= 80 else "\U0001f7e1" if s >= 50 else "\U0001f534"
        st = r.get("stats", {})
        n_sz = st.get("total_seizure_events", st.get("total_seizures", "\u2014"))
        ictal_ready = "\u2714" if st.get("has_seizure_annotations") else "\u2718"
        rows.append([light, r["name"], r["type"], f"{s}/100",
                     len(r.get("issues", [])), len(r.get("warnings", [])),
                     n_sz, ictal_ready])
    print(tabulate(rows, headers=["", "Dataset", "Tipo", "Score", "Issues",
                                   "Warns", "Crisi", "Ictal-ready"],
                   tablefmt="simple"))

    avg = sum(r["score"] for r in reports) / len(reports) if reports else 0
    n_ictal = sum(1 for r in reports if r.get("stats", {}).get("has_seizure_annotations"))
    print(f"\n  Score medio: {avg:.0f}/100")
    print(f"  Dataset con crisi annotate (usabili come test ictale): {n_ictal}/{len(reports)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"timestamp": datetime.now().isoformat(),
             "datasets_root": str(root.resolve()), "reports": reports},
            indent=2, default=str))
        print(f"\n  {ok(f'Report JSON salvato: {Path(args.json_out).resolve()}')}")

    sys.exit(1 if any(r.get("issues") for r in reports) else 0)


if __name__ == "__main__":
    main()