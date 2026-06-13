<div align="center">

# eegpipe

### A Production-Grade EEG / BCI Machine-Learning Pipeline

*From exploratory notebooks → a modular, reproducible, deployable deep-learning system for EEG.*

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE--Python-1.6%2B-00897B)
![Hydra](https://img.shields.io/badge/config-Hydra-89CFF0)
![MLflow](https://img.shields.io/badge/tracking-MLflow%20%7C%20W%26B-0194E2)
![DVC](https://img.shields.io/badge/data-DVC-13ADC7?logo=dvc&logoColor=white)
![FastAPI](https://img.shields.io/badge/serving-FastAPI%20%7C%20ONNX-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen)
![Lint](https://img.shields.io/badge/lint-ruff-000000)
![License](https://img.shields.io/badge/license-MIT-yellow)

</div>

---

## Table of contents

- [Overview](#overview)
- [Capability matrix](#capability-matrix)
- [System architecture](#system-architecture)
- [The ML pipeline](#the-ml-pipeline)
- [Developer workflow](#developer-workflow)
- [Leakage-safe evaluation](#leakage-safe-evaluation)
- [Sleep-staging model internals](#sleep-staging-model-internals)
- [Self-supervised pretraining & transfer](#self-supervised-pretraining--transfer)
- [Serving & deployment](#serving--deployment)
- [Results & verified outputs](#results--verified-outputs)
- [Quickstart](#quickstart)
- [Configuration (Hydra)](#configuration-hydra)
- [CLI reference](#cli-reference)
- [Project layout](#project-layout)
- [Testing & quality](#testing--quality)
- [License](#license)

---

## Overview

`eegpipe` is a refactor of the original `eeganalysis` notebooks into an installable, tested,
production-ready package. It spans the **entire EEG/BCI lifecycle**: BIDS/MOABB/Sleep-EDF
ingestion → leakage-safe cross-validation → classical (Riemannian) **and** deep
(EEGNet / Conformer / ST-GCN) decoding → sleep-staging sequence models
(BiLSTM / Transformer / U-Time) with self-supervised pretraining and foundation-model transfer
→ interpretability and cross-site domain adaptation → ONNX export, a hardened FastAPI service,
and Triton / KServe / Kubernetes deployment — all under Hydra configs, DVC data versioning,
experiment tracking, unit tests, and CI.

> **Design thesis:** EEG/BCI projects fail in production for predictable reasons — *subject
> data leakage*, configs trapped in notebooks, no experiment provenance, and raw recordings
> committed to git. Every directory here is shaped to make those mistakes *hard to commit*.

---

## Capability matrix

| Domain | What's included | Module |
|---|---|---|
| **Data** | BIDS · MOABB (BCI IV-2a) · Sleep-EDF · MNE-sample, behind one `EpochsBundle` | `data/loaders.py` |
| **Leakage safety** | Subject-aware `GroupKFold` / LOSO / nested CV + leakage assertions | `data/splits.py` |
| **Preprocessing** | Band-pass · notch · resample · **automated ICA (ICLabel)** · AutoReject | `preprocessing/` |
| **Classical features** | **Riemannian** tangent space · wavelets · spectral band-power | `features/` |
| **Deep models** | EEGNet · EEG-Conformer · ST-GCN | `models/` |
| **Sleep sequence models** | TinySleepNet (BiLSTM) · SleepTransformer · U-Time | `models/sleep.py` |
| **Self-supervised** | Relative Positioning pretext + encoder transfer | `models/ssl.py` |
| **Foundation models** | BENDR / LaBraM adapter into the shared encoder slot | `models/foundation.py` |
| **Domain adaptation** | Euclidean Alignment · Riemannian recentering (cross-site) | `adaptation/` |
| **Interpretability** | Integrated-Gradients saliency · attention rollout · confusion matrix | `interpret/` |
| **Real-time** | Lab Streaming Layer (LSL) closed-loop inference | `inference/stream.py` |
| **Serving** | FastAPI (ONNX-first, hardened) · Streamlit dashboard | `serving/`, `dashboard/` |
| **MLOps** | Hydra · MLflow/W&B · DVC pipeline · CI regression gate | `configs/`, `dvc.yaml`, CI |
| **Deployment** | Docker · k8s + HPA · KServe · Triton | `docker/`, `deploy/` |

---

## System architecture

```mermaid
flowchart LR
    subgraph SRC["Data sources"]
        direction TB
        BIDS["EEG-BIDS"]
        MOABB["MOABB · BCI IV-2a"]
        SEDF["Sleep-EDF"]
        SAMP["MNE sample"]
    end

    subgraph CORE["eegpipe core"]
        direction TB
        LOAD["data.loaders<br/>Factory → EpochsBundle"]
        PRE["preprocessing<br/>filter · ICA · AutoReject"]
        SPLIT["data.splits<br/>subject-aware CV"]
        ADAPT["adaptation<br/>Euclidean / Riemannian align"]
        FEAT["features<br/>Riemann · wavelet · spectral"]
        subgraph MZ["models"]
            direction TB
            ENC["EpochEncoder · shared"]
            DEEP["EEGNet · Conformer · ST-GCN"]
            SEQ["TinySleepNet · Transformer · U-Time"]
            SSL["SSL · Relative Positioning"]
            FND["Foundation adapter · BENDR/LaBraM"]
        end
        TRAIN["training<br/>Lightning loops"]
        INTERP["interpret<br/>saliency · attention · CM"]
    end

    subgraph OPS["MLOps"]
        direction TB
        HYDRA["Hydra configs"]
        DVC["DVC · data versioning"]
        MLF["MLflow / W&B"]
        GATE["CI regression gate"]
    end

    subgraph OUT["Serving & apps"]
        direction TB
        EXP["ONNX / TorchScript export"]
        API["FastAPI · /predict"]
        LSL["LSL real-time loop"]
        DASH["Streamlit dashboard"]
    end

    subgraph DEP["Deployment"]
        direction TB
        DOCK["Docker"]
        K8S["k8s + HPA"]
        KS["KServe"]
        TRT["Triton"]
    end

    SRC --> LOAD --> PRE --> SPLIT
    SPLIT --> FEAT --> MZ
    SPLIT --> MZ
    ADAPT --> MZ
    MZ --> TRAIN --> INTERP
    TRAIN --> EXP
    EXP --> API & LSL & DASH
    EXP --> DEP

    HYDRA -. config .-> CORE
    DVC -. versions .-> LOAD
    TRAIN -. metrics .-> MLF
    MLF -. baseline .-> GATE

    classDef src fill:#e3f2fd,stroke:#1976d2,color:#0d47a1;
    classDef core fill:#ede7f6,stroke:#5e35b1,color:#311b92;
    classDef ops fill:#fff3e0,stroke:#fb8c00,color:#e65100;
    classDef out fill:#e8f5e9,stroke:#43a047,color:#1b5e20;
    classDef dep fill:#fce4ec,stroke:#d81b60,color:#880e4f;
    class BIDS,MOABB,SEDF,SAMP src;
    class LOAD,PRE,SPLIT,ADAPT,FEAT,ENC,DEEP,SEQ,SSL,FND,TRAIN,INTERP core;
    class HYDRA,DVC,MLF,GATE ops;
    class EXP,API,LSL,DASH out;
    class DOCK,K8S,KS,TRT dep;
```

---

## The ML pipeline

End-to-end data flow, from raw recording to a served prediction. Stateful steps (ICA,
AutoReject, scalers) are **fit inside each CV fold** — never on the test subject.

```mermaid
flowchart TD
    A["Raw EEG<br/>(.fif / .edf / BIDS)"] --> B["Load → EpochsBundle<br/>X (n,C,T) · y · subject_ids"]
    B --> C["Preprocess<br/>band-pass · notch · resample"]
    C --> D{"Artefacts?"}
    D -->|yes| E["ICA + ICLabel<br/>AutoReject"]
    D -->|no| F
    E --> F["Epoch sequences<br/>(sleep) or trials"]
    F --> G[["Subject-aware split<br/>GroupKFold / LOSO"]]
    G --> H["Domain alignment<br/>Euclidean / Riemannian"]

    H --> I1["Classical path<br/>Covariances → Tangent space"]
    H --> I2["Deep path<br/>EEGNet · Conformer · sequence nets"]
    I1 --> J["Cross-validated metrics<br/>κ · macro-F1 · balanced-acc"]
    I2 --> J
    J --> K{"Beats baseline<br/>+ regression gate?"}
    K -->|no| X["Iterate: features /<br/>architecture / SSL / data"]
    X --> C
    K -->|yes| L["Export ONNX / TorchScript"]
    L --> M["FastAPI · Triton · KServe<br/>+ Streamlit / LSL"]

    classDef gate fill:#ffebee,stroke:#c62828,color:#b71c1c,font-weight:bold;
    classDef art fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef proc fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    class G,K gate;
    class A,B,L art;
    class C,E,F,H,I1,I2,J,M proc;
```

---

## Developer workflow

Everything is a composable command. A run is fully specified by *config + seed + data hash*,
so any result is reproducible.

```mermaid
flowchart LR
    S(["Start"]) --> INST["make install-dev<br/>editable + extras + hooks"]
    INST --> CFG["Compose Hydra config<br/>data × preprocess × model × training"]
    CFG --> DEMO["make demo<br/>end-to-end smoke"]
    DEMO --> DEC{"Task?"}

    DEC -->|decode / BCI| TR1["eegpipe-train"]
    DEC -->|sleep staging| TR2["eegpipe-sleep"]
    DEC -->|few labels| PRE["eegpipe-pretrain<br/>(SSL) → fine-tune"]
    DEC -->|baseline| EV["eegpipe-eval<br/>(Riemannian + align)"]

    TR1 & TR2 & PRE & EV --> TRACK["Track<br/>MLflow / W&B + DVC"]
    TRACK --> GATE{"Regression<br/>gate passes?"}
    GATE -->|no| CFG
    GATE -->|yes| EXP["eegpipe-export → ONNX"]
    EXP --> SRV["make serve / deploy"]
    SRV --> E(["Production"])

    classDef cmd fill:#ede7f6,stroke:#5e35b1,color:#311b92;
    classDef gate fill:#fff3e0,stroke:#ef6c00,color:#e65100,font-weight:bold;
    class INST,CFG,DEMO,TR1,TR2,PRE,EV,TRACK,EXP,SRV cmd;
    class DEC,GATE gate;
```

---

## Leakage-safe evaluation

The single most common EEG-ML error is letting epochs from one subject land in both train and
test, inflating accuracy by 10–30 points. `eegpipe` exposes **only** group-aware splitters and
asserts disjointness inside every fold.

```mermaid
flowchart TB
    subgraph POOL["Subjects S1…S9 (BCI IV-2a)"]
        s1["S1"]; s2["S2"]; s3["S3"]; s4["S4"]; s5["S5"]
        s6["S6"]; s7["S7"]; s8["S8"]; s9["S9"]
    end
    POOL --> SPL["StratifiedGroupKFold / LeaveOneGroupOut"]
    SPL --> TR["TRAIN folds<br/>S1 S2 S4 S5 S7 S8"]
    SPL --> VA["VAL fold<br/>S3 S6"]
    SPL --> TE["TEST (held-out subjects)<br/>S9"]
    TR --> CHK[["assert_no_subject_leakage()"]]
    VA --> CHK
    TE --> CHK
    CHK --> OK["disjoint → fit preprocessing on TRAIN only"]
    CHK --> BAD["overlap → raises ValueError"]

    classDef good fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef bad fill:#ffebee,stroke:#c62828,color:#b71c1c;
    classDef chk fill:#fff8e1,stroke:#f9a825,color:#f57f17,font-weight:bold;
    class OK good; class BAD bad; class CHK chk;
```

---

## Sleep-staging model internals

A 30-second epoch is ambiguous alone — neighbouring epochs disambiguate it. All three sleep
backbones share one `EpochEncoder` and differ only in how they model context *across* epochs
(BiLSTM / self-attention / multi-scale U-Net). Many-to-many: a stage is predicted for **every**
epoch in the input sequence.

```mermaid
flowchart LR
    IN["Input sequence<br/>(B, L, C, T)<br/>L × 30s epochs"] --> ENC

    subgraph ENC["EpochEncoder (shared, per epoch)"]
        direction TB
        C1["Conv1d + BN + GELU"] --> P1["MaxPool + Dropout"]
        P1 --> C2["Conv1d ×2"] --> AP["AdaptiveAvgPool → (B,L,emb)"]
    end

    ENC --> CTX{"Sequence model"}
    CTX -->|TinySleepNet| L1["BiLSTM + residual skip"]
    CTX -->|SleepTransformer| L2["Positional enc + Self-attention ×N"]
    CTX -->|U-Time| L3["Temporal U-Net (down/up + skips)"]
    L1 & L2 & L3 --> HEAD["Per-epoch linear head<br/>→ (B, L, 5)"]
    HEAD --> HYP["Overlap-averaged inference<br/>→ Hypnogram"]

    SSLIN["Unlabelled nights"] -. Relative Positioning SSL .-> ENC
    FND["BENDR / LaBraM"] -. FoundationEncoderAdapter .-> ENC

    classDef enc fill:#ede7f6,stroke:#5e35b1,color:#311b92;
    classDef seq fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef io fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    class C1,P1,C2,AP enc; class L1,L2,L3,HEAD seq; class IN,HYP,SSLIN,FND io;
```

> **Class imbalance** (N2 dominates, N1 is rare) is countered three ways at once: inverse-frequency
> class weights, a weighted sampler that oversamples sequences containing rare stages, and an
> optional focal loss. Model selection uses **macro-F1**, never accuracy.

---

## Self-supervised pretraining & transfer

The defining constraint of clinical EEG is *few labels, lots of unlabelled signal*.

```mermaid
flowchart LR
    U["Unlabelled recordings"] --> RP["Sample epoch pairs<br/>close (≤τ⁺) vs far (≥τ⁻)"]
    RP --> PT["Train EpochEncoder<br/>'are these temporally close?'"]
    PT --> W["ssl_encoder.pt<br/>(pretrained weights)"]
    W --> INJ["load_pretrained_encoder()<br/>strict=False"]
    INJ --> FT["Fine-tune sleep model<br/>on few labelled nights"]
    FT --> METRIC["↑ macro-F1 in low-label regime"]

    classDef ssl fill:#e0f7fa,stroke:#00838f,color:#006064;
    class U,RP,PT,W,INJ,FT,METRIC ssl;
```

---

## Serving & deployment

**Request flow** (FastAPI, ONNX-first backend):

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as FastAPI
    participant K as API-key guard
    participant P as Predictor (ONNX/Torch)
    C->>A: POST /predict {data, sfreq}
    A->>A: timing middleware start
    A->>K: check X-API-Key
    alt invalid key
        K-->>C: 401 Unauthorized
    else ok
        A->>A: validate shape · size guard (413)
        A->>P: predict_proba(X)
        P-->>A: softmax probabilities
        A-->>C: 200 {prediction, label, probs,<br/>model_name/version, backend, inference_ms}
    end
    Note over A: GET /health (liveness) · /ready (readiness) · /version
```

**Deployment topology** — pick the rung you actually need:

```mermaid
flowchart TB
    CKPT["Trained checkpoint"] --> EXP["eegpipe-export → model.onnx"]
    EXP --> STORE[("Object store<br/>S3 / MinIO · DVC / MLflow registry")]
    STORE --> IMG["Docker image<br/>(CPU torch / ONNX runtime)"]

    IMG --> K8S
    IMG --> KS
    STORE --> TRT

    subgraph K8S["Kubernetes"]
        D["Deployment<br/>/ready · /health probes"] --> HPA["HPA · 2→10 pods @70% CPU"]
    end
    subgraph KS["KServe"]
        ISVC["InferenceService<br/>scale-to-zero · canary"]
    end
    subgraph TRT["Triton"]
        REPO["model_repository<br/>dynamic batching · GPU"]
    end

    K8S & KS & TRT --> CLIENTS["Clients · dashboard · LSL"]

    classDef store fill:#fff3e0,stroke:#fb8c00,color:#e65100;
    classDef dep fill:#fce4ec,stroke:#d81b60,color:#880e4f;
    class STORE store; class D,HPA,ISVC,REPO dep;
```

---

## Results & verified outputs

> All outputs below are **real** from this repo's verification runs. Deep-model accuracy
> numbers are intentionally **not** quoted as benchmarks here — they depend on your data/run
> and are emitted live to MLflow/`reports/`. The classical numbers shown are from the offline
> **synthetic** smoke (clearly labelled), used to prove the pipeline wiring, not to claim SOTA.

### Repository at a glance

| Metric | Value |
|---|---|
| Python modules (`src/`) | **51** |
| Hydra config files | **17** |
| Test files / CLI entry points | **11 / 6** |
| Deployment manifests | **7** |
| Unit tests | **22 passed**, 18 skipped (optional deps run in CI) |
| AST integrity check | **86 symbols across 32 modules — all present** |

### `make demo` — end-to-end smoke (synthetic, offline)

```text
[ok] data       synthetic X=(800, 3, 3000) subjects=4
[ok] windows    40 sequences of len 20
[ok] split      train=20 test=20 sequences, no subject leakage
[ok] align      per-subject mean-cov deviation from I = 2.61e-07
[ok] classical  macro_f1=0.990 kappa=0.987
[--] deep       torch/lightning not installed (runs in CI / on your machine)
[--] export     needs deep stage + onnx/onnxruntime
[--] hypnogram  needs deep stage (torch)
[ok] gate       [OK] test/macro_f1: baseline=0.0000 current=0.9896 (Δ=+0.9896)
SMOKE DEMO: 6 passed, 3 skipped, 0 failed
```

`reports/smoke_metrics.json` (synthetic baseline, written by the demo):

```json
{ "test/accuracy": 0.99, "test/balanced_accuracy": 0.990,
  "test/macro_f1": 0.990, "test/kappa": 0.987 }
```

### What each stage produces (and its verified status)

| Stage | Module | Output artefact | Verified |
|---|---|---|---|
| Load | `data/loaders.py` | `EpochsBundle` (X, y, subject_ids) | 4 loaders concrete |
| Preprocess | `preprocessing/` | cleaned MNE `Epochs` | pipeline builds & runs |
| Window | `data/sequence.py` | `(n, L)` index windows (no night-crossing) | boundary tests pass |
| Split | `data/splits.py` | disjoint train/val/test indices | leakage asserts pass |
| Align | `adaptation/` | whitened trials (mean-cov → I, dev 2.6e-7) | measured |
| Classical | `features/` | κ / macro-F1 (Riemannian / band-power) | macro-F1 0.99 (synthetic) |
| Deep train | `training/`, `models/` | `.ckpt` + MLflow run | CI / local (needs torch) |
| Interpret | `interpret/` | `reports/confusion.npy` · saliency · attention | plot + logic tested |
| Export | `serving/export.py` | `model.onnx` (dynamic axes) | CI (needs torch+onnx) |
| Serve | `serving/api.py` | `/predict` JSON + latency | health/ready/predict tested |
| Hypnogram | `inference/hypnogram.py` | per-epoch stages + plot | CI (needs torch) |

> **A real find from the real-data path.** Running `make demo REAL=--real` surfaced a bug that
> `py_compile` and import checks had *missed*: a save-time formatter had truncated
> `SleepEDFLoader.load`, leaving the class abstract (it still imported fine). It was fixed, and a
> repo-wide **AST integrity check** now guards against truncation. The PhysioNet download itself is
> firewalled in the build sandbox, so the `--real` path runs on your machine.

---

## Quickstart

```bash
# 1) Environment (Python ≥3.10)
python -m venv .venv && source .venv/bin/activate
make install-dev                     # editable install + all extras + pre-commit

# 2) Prove the whole chain in one command
make demo                            # synthetic, offline (deep stages skip without torch)
make demo REAL=--real                # one real Sleep-EDF night (downloads via MNE)

# 3) Train
eegpipe-train data=bci_iv_2a model=eegnet training.logger=mlflow      # motor-imagery, LOSO-ready
eegpipe-sleep data=sleep_edf model=sleep_transformer training=sleep preprocess=sleep

# 4) Few labels? Pretrain self-supervised, then fine-tune
eegpipe-pretrain data=sleep_edf model=sleep_seqnet training=pretrain preprocess=sleep
eegpipe-sleep    data=sleep_edf model=sleep_seqnet training=sleep    preprocess=sleep \
    model.pretrained_encoder=models_store/ssl_encoder.pt

# 5) Export & serve
eegpipe-export sleep.ckpt sleep.onnx --kind sequence --channels 4 --times 3000 --seq-len 20
EEGPIPE_ONNX=sleep.onnx make serve   # FastAPI → http://localhost:8000/docs
make dashboard                       # Streamlit → http://localhost:8501
```

---

## Configuration (Hydra)

Compose any run from four config groups — no code edits, fully logged:

```bash
eegpipe-<cmd>  data=<…>  preprocess=<…>  model=<…>  training=<…>  [overrides]
```

| Group | Options | Notes |
|---|---|---|
| `data` | `bci_iv_2a` · `sleep_edf` · `sleep_edf_multimodal` · `bids` · `mne_sample` | multimodal adds EOG+EMG (helps REM) |
| `preprocess` | `default` (8–32 Hz) · `sleep` (0.3–35 Hz) | sleep keeps delta/slow-wave bands |
| `model` | `eegnet` · `conformer` · `stgcn` · `sleep_seqnet` · `sleep_transformer` · `utime` · `sleep_foundation` | foundation = BENDR/LaBraM slot |
| `training` | `default` · `sleep` · `pretrain` | sleep selects on macro-F1, weighted sampler |

```bash
# Examples of composition + overrides
eegpipe-sleep data=sleep_edf_multimodal model=utime training=sleep preprocess=sleep
eegpipe-eval  data=bci_iv_2a loso=true align=true            # LOSO + Euclidean Alignment
eegpipe-train model=conformer model.depth=6 training.fast_dev_run=true
eegpipe-train -m model=eegnet,conformer model.lr=1e-4,1e-3   # Optuna/Hydra sweep
```

---

## CLI reference

| Command | Purpose |
|---|---|
| `eegpipe-train` | Train a per-trial decoder (EEGNet/Conformer/ST-GCN), subject-aware CV |
| `eegpipe-sleep` | Train a sleep-staging sequence model (macro-F1 selection) |
| `eegpipe-pretrain` | Self-supervised (Relative Positioning) encoder pretraining |
| `eegpipe-eval` | Leakage-safe Riemannian baseline (+ optional alignment) |
| `eegpipe-export` | Export a checkpoint to ONNX / TorchScript |
| `eegpipe-serve` | Launch the FastAPI inference service |
| `make demo` | End-to-end smoke test (`REAL=--real` for Sleep-EDF) |
| `make dashboard` | Streamlit app (Signals · Topomap · Predict · Hypnogram · Explain) |
| `make regression` | Fail if metrics regressed vs `tests/baseline_metrics.json` |

---

## Project layout

```
eeganalysis/
├── README.md  ROADMAP.md            # this file · the §1–§10 architecture plan
├── pyproject.toml  Makefile         # packaging (extras: dl,features,mlops,serve,export,dev)
├── configs/                         # Hydra groups: data · preprocess · model · training
├── src/eegpipe/
│   ├── data/         loaders (Factory) · sequence windows · subject-aware splits
│   ├── preprocessing/ filters · ICA+ICLabel · AutoReject · pipeline builder
│   ├── features/     riemann · wavelet · spectral
│   ├── adaptation/   Euclidean / Riemannian domain alignment
│   ├── models/       eegnet · conformer · stgcn · sleep · ssl · foundation · base · registry
│   ├── training/     datamodule · train · evaluate · sleep · pretrain
│   ├── inference/    predictor · onnx_predictor · stream (LSL) · hypnogram
│   ├── interpret/    saliency · attention · plots
│   ├── serving/      api (FastAPI) · schemas · export
│   ├── dashboard/    app (Streamlit, 5 tabs)
│   └── utils/        config · logging · seed · metrics
├── scripts/          thin CLI wrappers + smoke_demo + check_regression
├── tests/            pytest (subject-leakage tests are first-class)
├── docker/  deploy/  Dockerfile · compose · k8s + HPA · KServe · Triton
├── dvc.yaml params.yaml             # reproducible data → features → train → eval DAG
└── .github/workflows/ci.yml         # lint · type · test · export-smoke · regression-gate · docker
```

---

## Testing & quality

```bash
make test      # fast unit tests (excludes slow/gpu)
make cov       # full suite + coverage
make lint      # ruff
make type      # mypy
```

- **Subject-leakage tests are first-class** — a grep for `KFold` outside `splits.py` should return nothing.
- Tests needing `torch` / `onnx` / `pyriemann` **skip gracefully** locally and **run in CI**.
- CI (`.github/workflows/ci.yml`): ruff → mypy → pytest+coverage → export import-smoke →
  **model-regression gate** → Docker build.

---

## License

MIT © contributors. See [`ROADMAP.md`](ROADMAP.md) for the full senior-to-junior architecture
review (§1 architecture → §10 end-to-end smoke).

<div align="center"><sub>Built with MNE-Python · PyTorch Lightning · Hydra · MLflow · DVC · FastAPI · ONNX · Streamlit</sub></div>
