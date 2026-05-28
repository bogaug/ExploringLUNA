---
license: cc-by-nd-4.0
language:
- en
library_name: pytorch
tags:
- eeg
- biosignal
- time-series
- cross-attention
- foundation-model
- self-supervised
- masked-modeling
- transformer
- rope
- neuroscience
- neurips-2025
datasets:
- TUEG
- Siena
- TUAB
- TUAR
- TUSL
- SEED-V
metrics:
- balanced_accuracy
- roc_auc
- pr_auc
- f1
- cohen_kappa
thumbnail: https://raw.githubusercontent.com/pulp-bio/BioFoundation/refs/heads/main/docs/model/logo/LUNA_logo.png
model-index:
  - name: LUNA-Base
    results:
      - task:
          type: time-series-classification
          name: EEG Abnormality Detection
        dataset:
          type: TUAB
          name: TUH EEG Abnormal Corpus (TUAB)
        metrics:
          - type: balanced_accuracy
            value: 80.63
            name: Balanced Accuracy (%)
          - type: roc_auc
            value: 0.8868
            name: AUROC
          - type: pr_auc
            value: 0.8953
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Artifact Detection
        dataset:
          type: TUAR
          name: TUH EEG Artifact Corpus (TUAR)
        metrics:
          - type: roc_auc
            value: 0.902
            name: AUROC
          - type: pr_auc
            value: 0.495
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Slowing Classification
        dataset:
          type: TUSL
          name: TUH EEG Slowing Corpus (TUSL)
        metrics:
          - type: roc_auc
            value: 0.767
            name: AUROC
          - type: pr_auc
            value: 0.301
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Emotion Recognition
        dataset:
          type: SEED-V
          name: SEED-V
        metrics:
          - type: balanced_accuracy
            value: 37.30
            name: Balanced Accuracy (%)
          - type: cohen_kappa
            value: 0.1831
            name: Cohen's Kappa
          - type: f1
            value: 0.3389
            name: Weighted F1
            config: weighted
  - name: LUNA-Large
    results:
      - task:
          type: time-series-classification
          name: EEG Abnormality Detection
        dataset:
          type: TUAB
          name: TUH EEG Abnormal Corpus (TUAB)
        metrics:
          - type: balanced_accuracy
            value: 80.96
            name: Balanced Accuracy (%)
          - type: roc_auc
            value: 0.8924
            name: AUROC
          - type: pr_auc
            value: 0.8986
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Artifact Detection
        dataset:
          type: TUAR
          name: TUH EEG Artifact Corpus (TUAR)
        metrics:
          - type: roc_auc
            value: 0.918
            name: AUROC
          - type: pr_auc
            value: 0.505
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Slowing Classification
        dataset:
          type: TUSL
          name: TUH EEG Slowing Corpus (TUSL)
        metrics:
          - type: roc_auc
            value: 0.771
            name: AUROC
          - type: pr_auc
            value: 0.293
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Emotion Recognition
        dataset:
          type: SEED-V
          name: SEED-V
        metrics:
          - type: balanced_accuracy
            value: 39.18
            name: Balanced Accuracy (%)
          - type: cohen_kappa
            value: 0.2073
            name: Cohen's Kappa
          - type: f1
            value: 0.3586
            name: Weighted F1
            config: weighted
  - name: LUNA-Huge
    results:
      - task:
          type: time-series-classification
          name: EEG Abnormality Detection
        dataset:
          type: TUAB
          name: TUH EEG Abnormal Corpus (TUAB)
        metrics:
          - type: balanced_accuracy
            value: 81.57
            name: Balanced Accuracy (%)
          - type: roc_auc
            value: 0.8957
            name: AUROC
          - type: pr_auc
            value: 0.9029
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Artifact Detection
        dataset:
          type: TUAR
          name: TUH EEG Artifact Corpus (TUAR)
        metrics:
          - type: roc_auc
            value: 0.921
            name: AUROC
          - type: pr_auc
            value: 0.528
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Slowing Classification
        dataset:
          type: TUSL
          name: TUH EEG Slowing Corpus (TUSL)
        metrics:
          - type: roc_auc
            value: 0.802
            name: AUROC
          - type: pr_auc
            value: 0.289
            name: AUC-PR
      - task:
          type: time-series-classification
          name: EEG Emotion Recognition
        dataset:
          type: SEED-V
          name: SEED-V
        metrics:
          - type: balanced_accuracy
            value: 39.00
            name: Balanced Accuracy (%)
          - type: cohen_kappa
            value: 0.2037
            name: Cohen's Kappa
          - type: f1
            value: 0.3506
            name: Weighted F1
            config: weighted
---


<div align="center">
  <img src="https://raw.githubusercontent.com/pulp-bio/BioFoundation/refs/heads/main/docs/model/logo/LUNA_logo.png" alt="LUNA Logo" width="800"/>
  <h1>LUNA: Efficient and Topology-Agnostic Foundation Model for EEG</h1>
</div>
<p align="center">
  <a href="https://github.com/pulp-bio/BioFoundation">
    <img src ="https://img.shields.io/github/stars/pulp-bio/BioFoundation?color=ccf" alt="Github">
  </a>
  <a href="https://creativecommons.org/licenses/by-nd/4.0/">
    <img src="https://img.shields.io/badge/License-CC_BY--ND_4.0-lightgrey.svg" alt="License">
  </a>
  <a href="https://arxiv.org/abs/2510.22257">
    <img src="https://img.shields.io/badge/arXiv-2510.22257-b31b1b.svg" alt="Paper">
  </a>
</p>

**LUNA** (Latent Unified Network Architecture) is a **self-supervised foundation model for EEG** that makes models **agnostic to electrode topology**. LUNA projects arbitrary channel layouts into a **fixed-size latent space with learned queries + cross-attention**, then runs **patch-wise temporal self-attention** only on this compact latent. This **decouples compute from channel count**, yielding **linear-in-channels scaling**, large FLOPs/memory savings, and strong transfer across datasets and montages.

---

## 🔒 License & Usage Policy (Weights)

**Weights license:** The released model weights are licensed under **Creative Commons Attribution–NoDerivatives 4.0 (CC BY-ND 4.0)**. This section summarizes the practical implications for users. *This is not legal advice; please read the full license text.*

### ✅ You may
- **Use** and **redistribute** the **unmodified** LUNA weights (including in commercial settings) **with proper attribution** to the LUNA authors.
- **Fine-tune / adapt** the weights **for your internal use** (research or production) **without redistributing** the modified weights.
- **Publish your code, configs, logs, and papers** describing experiments with LUNA (please cite the paper).

### 🚫 You may not
- **Share, host, or redistribute any modified weights** (including LoRA/adapter/delta checkpoints or pruned/quantized variants). Any parameter set that encodes an adaptation is considered a derivative and cannot be shared under CC BY-ND 4.0.
- **Imply endorsement** by the LUNA authors for any derivative or evaluation without our written permission.
- **Use the LUNA name** in a way that suggests your modified model is an official LUNA release.

### 🤝 How to contribute improvements (PR-gated releases)
We welcome community improvements via a **pull-request (PR)** workflow. If you believe your improvements should become an **official LUNA release**:
1. **Open a PR** in the [BioFoundation repository](https://github.com/pulp-bio/BioFoundation) describing the change (architecture/head/training recipe, datasets, preprocessing, compute).
2. Include **reproducibility artifacts**: configs, seeds, scripts, environment details, training/validation logs, and the **evaluation protocol** (e.g., TUAB/TUAR/TUSL) with exact splits.
3. Provide **comprehensive results** (AUROC/AUPR/BA, FLOPs, memory) vs. the baselines reported in the LUNA paper.
4. After **maintainer review**, approved changes will be **retrained/validated** and, if accepted, **released by the maintainers** as a new **official LUNA** checkpoint under **CC BY-ND 4.0**.

> Rationale: CC BY-ND protects users from fragmented, lower-quality “LUNA variants,” while still enabling internal fine-tuning and a path for the community to upstream improvements through review.

---

## 🔎 Model Summary

- **Goal:** Topology-agnostic EEG modeling with **linear-in-channels** compute/memory.
- **Core idea:** **Channel-Unification Module** uses **learned queries** (Q) with **cross-attention** to map any set of channels to a fixed latent; **temporal Transformer** then operates on that latent sequence.
- **Pre-training data:** TUEG + Siena, **>21,000 hours** of raw EEG; downstream subjects removed to avoid leakage.
- **Downstream tasks:** TUAB (abnormal), **TUAR** (artifacts), **TUSL** (slowing), **SEED-V** (emotion; unseen 62-ch montage).

---

## 🚀 Model Variants

| Variant | Parameters |
| :--- | ---: |
| **LUNA-Base** | **7M** |
| **LUNA-Large** | **43M** |
| **LUNA-Huge** | **311M** |

*Scaling increases depth/width of the temporal encoder and the query/embedding sizes in the unification module.* 

### ⚙️ Model size configs (ready-made YAMLs)

Pick a LUNA size by selecting one of the provided model configs:

- `config/model/LUNA_base.yaml` — Base (≈7M)  
- `config/model/LUNA_large.yaml` — Large (≈43M)  
- `config/model/LUNA_huge.yaml` — Huge (≈311M)

**Use it via experiment defaults override** (recommended):

```yaml
# inside config/experiment/LUNA_finetune.yaml
defaults:
  - override /data_module: finetune_data_module   # or subject_independent_data_module
  - override /model: LUNA_base                    # change to LUNA_large or LUNA_huge
  - override /scheduler: cosine
  - override /task: finetune_task_LUNA
  - override /criterion: finetune_criterion
```

**Or from the CLI** (no file edits):

```bash
python -u run_train.py +experiment=LUNA_finetune /model=LUNA_large
```

---

## 📊 Results (Highlights)

- **TUAR (artifact detection):** **AUROC 0.921** (LUNA-Huge).
- **TUSL (slowing, 4-class):** **AUROC 0.802** (LUNA-Huge).
- **TUAB (abnormal vs normal):** **Bal. Acc. 81.57%**, **AUROC 0.8957** (LUNA-Huge).

**Efficiency:** Up to **300× fewer FLOPs** and **≈10× lower GPU memory** vs quadratic spatio-temporal attention on dense caps / long windows, thanks to unifying channels **before** temporal attention. 

---

## 🧠 Intended Use & Limitations

**Intended use.** Research on EEG representation learning & classification (abnormality, artifacts, slowing, emotion), especially when **montages vary** or **channel counts are high**.

**Limitations.**
- **Not a medical device.** Do **not** use for clinical decisions without proper validation & regulatory clearance.  
- **Unseen topologies:** Zero-shot transfer to **very different/dense** layouts (e.g., SEED-V) can underperform SOTA despite positive scaling; consider augmenting pre-training montage diversity and spatial encodings.
- **Distribution shifts:** Performance varies across cohorts, devices, and label protocols; validate locally and consider domain adaptation.

---

## 🏗️ Architecture & Training

**Tokenizer & features.** EEG is patch-segmented; temporal features via 1D conv w/ GroupNorm+GELU; **frequency features** (FFT mag/phase → MLP) are added; 3D electrode coordinates encoded via **NeRF-style sinusoids → MLP** (positional enc).

**Channel-Unification Module.** **Q learned queries** cross-attend to **channel-wise patch features** to produce a **fixed Q×E latent** per patch; FFN + Transformer layers refine the query tokens. Complexity is **O(Q·C)** (linear in channels).

**Temporal encoder.** **Patch-wise Transformer** with **RoPE** operates on the latent sequence (length = #patches), **not** on channels×patches, reducing sequence length and cost substantially.

**Pre-training objective.** **Masked-patch reconstruction** with Smooth-L1; decoder uses **channel-indexed queries** to reconstruct masked tokens. **Query specialization loss** encourages diverse query–channel affinities. 

---

## 🔧 Fine-tuning — General Checklist

0. **Install & read data prep**: clone the [BioFoundation repo](https://github.com/pulp-bio/BioFoundation), set up the environment as described there, then open `make_datasets/README.md` for dataset-specific notes (naming, expected folder layout, and common pitfalls).
1. **Choose model size**: set `- override /model: {LUNA_base|LUNA_large|LUNA_huge}` in your experiment YAML (or `/model=...` via CLI).
2. **Point to weights**: set `pretrained_safetensors_path: /path/to/LUNA_*.safetensors` in the experiment YAML.
3. **Pick data module**:
   - **TUH datasets (TUAB/TUSL/TUAR)** → `- override /data_module: finetune_data_module` and optionally override `data_module.train/val/test.hdf5_file` paths.
   - **Non-TUH (e.g., SEED-V)** → `- override /data_module: subject_independent_data_module` and remove the TUH-specific `data_module` block.
4. **Task settings**: set `classification_type` (`bc`, `mc`, `mmc`, `mcc`) and `model.num_classes` to match your downstream task.
5. **Env vars**: export `DATA_PATH` (dataset root) and `CHECKPOINT_DIR` (artifacts).
6. **Trainer/optimizer**: adjust `gpus/devices`, `batch_size`, `max_epochs`, LR/scheduler if needed.
7. **I/O**: set `io.base_output_path` and confirm `io.checkpoint_dirpath` exists.

---

## 🧪 Example: Fine-tune on TUSL (end-to-end)

**0) Install & acquire data**  
- Follow the installation instructions in the [BioFoundation repository](https://github.com/pulp-bio/BioFoundation).  
- Read `make_datasets/README.md` for exact dataset preparation details.  
- Download the **raw TUSL** dataset from the official [TUH EEG corpus source](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml) and place it locally, e.g.: `/eeg_data/TUSL/`.

**1) Prepare data**

```bash
python make_datasets/process_raw_eeg.py tusl       --root_dir /eeg_data/TUSL/edf       --output_dir /processed_eeg

python make_datasets/make_hdf5.py       --prepath /processed_eeg       --dataset TUSL --remove_pkl
```

**2) Set environment variables**

```python
# run_train.py (example)
import os
os.environ["DATA_PATH"] = "/processed_eeg"   # contains TUSL_data/{train,val,test}.h5
os.environ["CHECKPOINT_DIR"] = "/LUNA_runs"  # directory for checkpoints & logs
```

**3) Edit the experiment file: `config/experiment/LUNA_finetune.yaml`**

```yaml
defaults:
  - override /data_module: finetune_data_module # Change based on dataset, finetune_data_module for TUH and subject_independent_data_module for non-TUH
  - override /model: LUNA_base              # Pick the model size, here base, but also available are LUNA_large / LUNA_huge.

pretrained_safetensors_path: /path/to/LUNA_base.safetensors

classification_type: "mcc" # Set based on what type of classification task (Multiclass Classification (MCC), Binary (BC), etc.)
model:
  num_classes: 4 # Set based on how many classes are in your dataset

# Write paths to preprocessed .h5 TUSL files
data_module:
  train:
    _target_: datasets.tuh_dataset.TUH_Dataset
    hdf5_file: ${env:DATA_PATH}/TUSL_data/train.h5 #Here point to the correct file
    finetune: true
  val:
    _target_: datasets.tuh_dataset.TUH_Dataset
    hdf5_file: ${env:DATA_PATH}/TUSL_data/val.h5 #Here point to the correct file
    finetune: true
  test:
    _target_: datasets.tuh_dataset.TUH_Dataset
    hdf5_file: ${env:DATA_PATH}/TUSL_data/test.h5 #Here point to the correct file
    finetune: true
```

**4) Launch**

```bash
python -u run_train.py +experiment=LUNA_finetune
```

*Tip*: to switch sizes without editing the file:

```bash
python -u run_train.py +experiment=LUNA_finetune /model=LUNA_large pretrained_safetensors_path=/path/to/LUNA_large.safetensors
```

---

## ⚖️ Responsible AI, Risks & Biases

- **Clinical safety:** research-only; human oversight required.  
- **Bias & drift:** montage/device/population differences can induce shifts; validate and monitor.  
- **Artifacts & rare events:** robustness varies; use QC and task-appropriate preprocessing.

---

## 🔗 Sources

- **Code:** https://github.com/pulp-bio/BioFoundation  
- **Paper:** LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis (arxiv:2510.22257).

---

## 🔗 Related Models
   
   - **[LuMamba](https://huggingface.co/PulpBio/LuMamba)** — Extends LUNA's channel-unification to a linear-complexity Mamba backbone, with systematic analysis of LeJEPA for biosignal SSL.
   - **[FEMBA](https://huggingface.co/PulpBio/FEMBA)** — Bi-directional Mamba foundation model for EEG; the temporal backbone that LuMamba builds on.
   - **[TinyMyo](https://huggingface.co/PulpBio/TinyMyo)** — Tiny foundation model for flexible EMG signal processing at the edge.

## 📜 Citation

If you use LUNA, please cite:

```bibtex
@inproceedings{
  doner2025luna,
  title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
  author={Berkay D{\"o}ner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=uazfjnFL0G}
}
```

---

## 🛠️ Maintenance & Contact

- **Issues & support:** please open a GitHub issue in the BioFoundation repository.

---

## 🗒️ Changelog

- **v1.0:** Initial release of LUNA model card with task-specific checkpoints and instructions.