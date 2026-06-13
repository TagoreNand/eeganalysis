# eegpipe — Architecture & Scaling Roadmap

> A senior-to-junior review and staged plan to take `eeganalysis` from exploratory
> notebooks to a robust, reproducible, deployable EEG/BCI ML system.
>
> This document is paired with a **working scaffold** already written into the repo
> (`src/eegpipe/`, `configs/`, `docker/`, `dvc.yaml`, CI). Every recommendation below maps
> to a real file you can open, run, and extend. `TODO(roadmap-§X)` markers in the code
> point back to the relevant section here.

---

## 0. Current-state review

You have seven solid MNE-Python notebooks (now in `notebooks/legacy/`) covering the full
classical pipeline: BIDS I/O, epoching/evoked, epoch cleaning, time-frequency, sensor-space
decoding, and source estimation. That domain coverage is genuinely strong — the gap is
*engineering*, not neuroscience. As a code review, the load-bearing issues are:

| # | Issue (in the notebooks) | Why it bites in production | Fix (this scaffold) |
|---|--------------------------|----------------------------|---------------------|
| 1 | **Logic lives in cells**, copy-pasted between notebooks (`pick_types`, filter, reshape). | No reuse, no tests, silent divergence between notebooks. | Promote to `src/eegpipe/` package, import everywhere. |
| 2 | **`StratifiedKFold` on reshaped trials** (`temporal analysis.ipynb`). | The moment you have >1 subject this leaks identity across folds and inflates accuracy 10–30 pts. | `data/splits.py` — group-aware CV is the *only* CV exposed. |
| 3 | Hard-coded `matplotlib.use('Qt5Agg')`, `out_data/` paths. | Breaks on any headless server / CI / Docker. | Config-driven paths (Hydra); Agg backend in serving. |
| 4 | **Manual ICA component picking** by eye. | Not reproducible, not scalable to N subjects. | `preprocessing/steps.py::ICAArtifactRemoval` + ICLabel. |
| 5 | sklearn `LogisticRegression` on flattened trials only. | Leaves large accuracy on the table vs. Riemannian / deep models. | `features/riemann.py`, `models/` (EEGNet, Conformer, ST-GCN). |
| 6 | No experiment tracking, no data versioning, no tests. | Results aren't reproducible; raw `.fif` risks landing in git. | MLflow/W&B, DVC, pytest, pre-commit (block >500 kB commits). |

Guiding principles for everything that follows: **depend on abstractions** (callers see
`BaseDataLoader`, not "the BIDS code"); **make the wrong thing hard** (leakage, committing
raw data); **every run fully specified by config + seed + data hash**.

---

## 1. Code architecture & software engineering

### 1.1 Target repository structure

```
eeganalysis/
├── ROADMAP.md  README.md  pyproject.toml  Makefile
├── configs/                      # Hydra config groups — every run is composed here
│   ├── config.yaml               #   root: defaults list + global params
│   ├── data/ preprocess/ model/ training/
├── src/eegpipe/                  # the installable package (src-layout)
│   ├── data/         loaders (Strategy) · splits (subject-aware) · torch_dataset (Adapter)
│   ├── preprocessing/ base · steps · pipeline (Builder)
│   ├── features/     riemann · wavelet · spectral
│   ├── models/       base (LightningModule) · eegnet · conformer · stgcn · ssl · registry (Factory)
│   ├── training/     datamodule · train · evaluate (nested subject CV)
│   ├── inference/    predictor · stream (LSL real-time)
│   ├── serving/      api (FastAPI) · schemas (Pydantic)
│   ├── dashboard/    app (Streamlit)
│   └── utils/        config · logging · seed
├── scripts/          thin CLI wrappers
├── tests/            pytest (subject-leakage tests are first-class)
├── docker/           Dockerfile + compose (api + mlflow + dashboard)
├── dvc.yaml params.yaml           # reproducible data→features→train→eval DAG
└── .github/workflows/ci.yml
```

**Why `src/` layout.** It forces you to test the *installed* package, not the local
working tree — catching "works on my machine because of CWD" bugs before CI does.

### 1.2 Migrating off notebooks (without losing them)

Notebooks stay — but demoted to *consumers* of the package, not authors of logic:

1. Extract each reusable cell into a package function/class (done for the headline paths).
2. Replace the cell body with a one-line call, e.g.
   `bundle = build_loader(cfg.data).load()`.
3. Keep notebooks for narrative/teaching in `notebooks/`; run them in CI with
   **papermill** + **nbstripout** (already a pre-commit hook) so outputs never bloat git.
4. Anything that must be a figure for a paper becomes a script under `scripts/` writing to
   `reports/`, so it's reproducible from the CLI.

### 1.3 Design patterns (and where each lives)

- **Strategy** — `data/base.py::BaseDataLoader`. BIDS, MOABB, Sleep-EDF, MNE-sample are
  interchangeable strategies returning one `EpochsBundle`. Callers never branch on dataset.
- **Factory + Registry** — `data/registry.py`, `models/registry.py`. `@register_loader("bci_iv_2a")`
  and `build_loader(cfg)` / `build_model(cfg, ...)` turn a config string into an object. Add
  a dataset/model by writing one class + one decorator; no edits to call sites.
- **Builder** — `preprocessing/pipeline.py::build_preprocessing`. Declarative YAML list of
  steps → a composed, fit/transform pipeline. Stateful steps (ICA, AutoReject) carry a
  `stateless=False` flag so the training loop knows to fit them **inside** each CV fold.
- **Template Method** — `models/base.py::LitEEGClassifier`. The train/val/test loop, loss,
  optimiser and logging are fixed; only `backbone.forward` varies. Swapping EEGNet→Conformer
  touches zero training code.
- **Adapter** — `EpochsBundle` (MNE↔arrays) and `EEGWindowsDataset` (arrays↔torch). The
  boundary between the neuroscience stack (MNE) and the DL stack (torch) is explicit and thin.
- **Dependency inversion via config** — Hydra composes `data × preprocess × model × training`
  from the CLI: `eegpipe-train data=sleep_edf model=conformer training.logger=mlflow`. The
  resolved config is logged with every run, so an experiment is fully reconstructable.

### 1.4 Engineering standards (wired into CI + pre-commit)

`ruff` (lint+format), `mypy` (types), `pytest` (+coverage), `pre-commit` (incl.
`check-added-large-files --maxkb=500` to stop a stray `.fif` reaching git, and `nbstripout`).
Console entry points (`eegpipe-train/eval/serve`) come from `pyproject.toml`. `make help`
lists every workflow.

---
## 2. Advanced ML & deep learning

### 2.1 Architectures (a ladder, not a single choice)

Climb this ladder; only move up when the rung below is genuinely saturated and you have the
data to justify the capacity.

| Model | File | Params | Use when | Notes |
|-------|------|--------|----------|-------|
| **Riemannian + LR** | `features/riemann.py` | tiny | *Always — the baseline.* | Near-SOTA on small MI sets; beats deep nets when trials are scarce. |
| **EEGNet** (2018) | `models/eegnet.py` | ~2–5k | Default deep baseline. | Depthwise conv ≈ learned spatial filters (CSP-like). Fast, robust. |
| **ShallowConvNet / DeepConvNet** | via `braindecode` | 10k–500k | Compare against EEGNet. | Use Braindecode's vetted implementations rather than re-rolling. |
| **EEG-Conformer** (2022) | `models/conformer.py` | ~150k+ | More data, need long-range temporal context. | Conv tokenizer → Transformer encoder. |
| **ST-GCN** | `models/stgcn.py` | varies | Sensor topology matters (source-ish, connectivity). | Graph built from montage 3-D positions (`adjacency_from_positions`). |
| **Foundation models** (BENDR, LaBraM) | `models/ssl.py` (transfer hook) | large | Cross-dataset transfer, label-scarce clinical EEG. | Fine-tune a pretrained encoder; see §2.4. |

Practical guidance: report **Riemannian vs EEGNet** first. If a 150k-param Conformer doesn't
beat a 2k-param EEGNet on your data, you have a *data* problem (augmentation, more subjects,
SSL), not an *architecture* problem. Don't skip rungs.

```python
# Swapping architectures is a config change, not a code change:
#   eegpipe-train model=eegnet
#   eegpipe-train model=conformer model.depth=6
#   eegpipe-train model=stgcn
```

### 2.2 Feature extraction beyond flattening

- **Riemannian geometry** (`RiemannianTangentSpace`). Trial covariance matrices are SPD and
  live on a curved manifold; projecting to the tangent space at the geometric mean linearises
  them so a plain logistic regression is extremely strong. This is the single highest-leverage
  upgrade to your `temporal analysis` notebook. Pair with **FBCSP** (filter-bank CSP) for MI.
- **Wavelets** (`WaveletEnergyFeatures`). DWT sub-band log-energy + entropy capture transient,
  non-stationary dynamics (ERP components, sleep spindles/K-complexes) that fixed-window FFT
  misses. Complements your `Frequency and time-frequency` notebook (which used Morlet TFR).
- **Spectral band-power** (`BandPowerFeatures`). Relative δ/θ/α/β/γ power — cheap, interpretable,
  the natural feature set for sleep staging and workload/affect tasks.
- **Cross-subject alignment** as a "feature" step: **Euclidean Alignment** or **Riemannian
  recentering** (whiten each subject's covariances by their reference mean) dramatically
  improves transfer. Add as a preprocessing/feature step fit per subject.

### 2.3 Automated artefact removal (kill the manual ICA)

`preprocessing/steps.py` provides:
- **`ICAArtifactRemoval`** — fits extended-Infomax ICA, then auto-labels components with
  **ICLabel** (`mne-icalabel`) and drops eye/muscle/heart/line-noise above a probability
  threshold. Falls back to EOG-correlation if ICLabel isn't installed.
- **`AutoRejectStep`** — learns per-channel peak-to-peak thresholds via the `autoreject`
  library (cross-validated), replacing hand-tuned `reject=dict(eeg=...)`.
- Roadmap extension: add **ASR** (Artefact Subspace Reconstruction) for online/streaming use,
  where ICA is too slow.

Crucial: ICA/AutoReject are **stateful** (`stateless=False`) — they must be `fit` on training
data only, *inside* each CV fold (`training/evaluate.py` shows the pattern). Fitting them on
the full dataset is a subtle but real leak.

### 2.4 Self-supervised & transfer learning for small datasets

EEG's defining constraint is *few labels, lots of unlabelled signal*. Exploit it:

- **Relative Positioning** (`models/ssl.py`, implemented). Pretext label = "are these two
  windows temporally close?" Trains an encoder on unlabelled recordings; transfer the encoder
  with `transfer_encoder(...)` and fine-tune a small head on your labelled task.
- **Contrastive SSL** (SeqCLR / SimCLR-style). Augment a window two ways (time-masking,
  channel-dropout, FT-surrogate, jitter) and pull the views together. Reuse the same encoder.
- **Masked autoencoding** — mask spans of channels/time and reconstruct; basis of modern EEG
  foundation models.
- **Foundation models** — fine-tune **BENDR** or **LaBraM** encoders for cross-dataset transfer.
  Hook: `transfer_encoder(pretrained_encoder, n_classes, emb_dim, freeze=True)`.
- **Data augmentation** as cheap regularisation (Braindecode's `augmentation` module): mixup,
  frequency-shift, sensor-rotation, smooth time-mask. Wire into `EEGWindowsDataset(transform=...)`.

### 2.5 Evaluation methodology (the part that makes results *true*)

Non-negotiables, all enforced in `training/`:
- **Subject-aware CV** (`GroupKFold` / `LeaveOneGroupOut`) — never plain KFold. LOSO is the
  headline number for "does it generalise to a new person?".
- **Nested CV** (`nested_subject_cv`) for unbiased HPO: tune on inner folds, report on the
  untouched outer test fold.
- **Metrics beyond accuracy**: Cohen's **κ** (BCI standard), **balanced accuracy** and
  **macro-F1** (mandatory for imbalanced sleep staging), plus confusion matrices logged to
  the tracker.
- **Statistics**: report mean ± std across folds; use a permutation/Wilcoxon test before
  claiming model A beats model B.

---
## 3. Standout components to add

### 3.1 Real-time streaming BCI (closed loop) — `inference/stream.py`

Consume a live **LSL** stream (OpenBCI, g.tec, or a replayed file via `mne-lsl`), maintain a
sliding window, and emit a prediction every `step_s` to a second LSL outlet that a game or
neurofeedback UI subscribes to. The same `Predictor` backs both the API and the stream — no
train/serve skew. Latency budget to respect: *window length + inference < your control loop*
(target < 250 ms for responsive MI). Roadmap extensions (marked in code): online artefact
gating (ASR), exponential smoothing of consecutive predictions, and a per-session calibration
phase with Riemannian recentering.

### 3.2 Sleep-stage classification — `data/sleep_edf.py` + a sequence head

Two things make sleep staging distinctive, and both are design decisions, not afterthoughts:
- **Temporal context across epochs.** A 30 s epoch is ambiguous in isolation; neighbouring
  epochs disambiguate it. Stack a sequence model (BiLSTM or Transformer, à la TinySleepNet /
  U-Time / SeqSleepNet) on top of per-epoch embeddings from EEGNet/Conformer. Architecturally:
  reuse the backbone as a per-epoch encoder, add a sequence module over windows of K epochs.
- **Severe class imbalance** (N2 dominates, N1 is rare). The scaffold already computes inverse-
  frequency `class_weights` in `training/train.py` and uses `StratifiedGroupKFold`; add
  macro-F1 as the early-stopping monitor so you don't optimise for "predict N2 always".

### 3.3 Interactive dashboard — `dashboard/app.py` (Streamlit)

Three tabs implemented (Signals/PSD, Topomap, Predict) plus a clear extension path:
- A **live LSL tab** that visualises `inference/stream.py` predictions in real time.
- An **explainability panel** — Grad-CAM / Integrated Gradients (Captum) over the input to
  show *which channels/time-points* drive a prediction. For a BCI/clinical tool, trust matters
  as much as accuracy; saliency maps are how clinicians sanity-check the model.
- Streamlit is the fast path; choose **Dash/Plotly** instead if you need fine-grained callback
  control or to embed in an existing web app. Both are fine — don't over-engineer the choice.

> These three are deliberately complementary: the dashboard is the *human* interface, the LSL
> loop is the *real-time* interface, and the FastAPI service (§4.3) is the *machine* interface —
> all three call the one `Predictor`.

---

## 4. MLOps, CI/CD & deployment

### 4.1 Experiment tracking — MLflow or W&B

Wired into `training/train.py::_make_logger` (`training.logger=mlflow|wandb|csv`).
- **MLflow** if you want a self-hosted, OSS server + **model registry** with stage transitions
  (Staging→Production) — the `docker-compose.yml` already stands one up on `:5000`.
- **W&B** if you want best-in-class sweep dashboards and team collaboration out of the box.
- **Log, every run**: the *resolved* Hydra config, per-fold metrics, confusion matrix, the
  data hash (DVC), git SHA, and the checkpoint. The bar: anyone can reconstruct the run from
  the tracker alone.
- **Hyper-parameter search**: Hydra's Optuna sweeper — `train.py::main` returns `test/acc`, so
  `eegpipe-train -m model.lr=1e-4,1e-3 model=eegnet,conformer hydra/sweeper=optuna` just works.

### 4.2 Data versioning — DVC (raw EEG never touches git)

`.fif`/`.edf`/`.set` are git-ignored; DVC tracks them and stores blobs in a remote (S3/MinIO/
GCS — see `.env.example`). The flow:

```bash
dvc init
dvc remote add -d storage s3://my-eeg-bucket/dvcstore
dvc add data/raw/                 # creates data/raw.dvc (tiny pointer, committed to git)
git add data/raw.dvc .gitignore && git commit -m "track raw EEG with DVC"
dvc push                          # blobs -> remote; teammates `dvc pull`
```

`dvc.yaml` + `params.yaml` define a reproducible **DAG** (raw → processed → train → eval).
`dvc repro` re-runs only the stages whose dependencies/params changed and records
`reports/metrics.json`, giving you data + code + metric lineage in one graph. For BIDS
datasets, version the BIDS root as one DVC-tracked directory. (Alternative at scale: lakeFS.)

### 4.3 Containerisation & serving — Docker + FastAPI

- **`serving/api.py`** — a typed FastAPI service: `GET /health`, `POST /predict` (single trial
  `(C,T)` or batch `(N,C,T)`), Pydantic schemas as the contract, checkpoint path from
  `EEGPIPE_CKPT` so one image serves any model. `uvicorn eegpipe.serving.api:app` or `make serve`.
- **`docker/Dockerfile`** — multi-stage, slim runtime, CPU-torch by default (one-line swap to a
  CUDA base for GPU), with a `/health` HEALTHCHECK.
- **`docker/docker-compose.yml`** — brings up API + MLflow + dashboard together for local dev.
- **Latency**: export the trained model to **TorchScript** or **ONNX Runtime** for serving
  (2–5× faster CPU inference, no Lightning/Python-class dependency at runtime). Add a
  `/predict/batch` path and micro-batching for throughput.
- **Scale-out** (later): Seldon Core / KServe on Kubernetes, or a Triton Inference Server if you
  consolidate many models. Don't reach for k8s until a single container is genuinely the bottleneck.
- **Monitoring**: log input feature distributions (band-power, covariance norms) and alert on
  **drift** — EEG is highly non-stationary across sessions/headsets, so production drift is the
  rule, not the exception.

### 4.4 CI/CD — `.github/workflows/ci.yml`

On every push/PR: ruff lint + format-check → mypy (advisory) → `pytest -m "not slow and not gpu"`
with coverage → Docker build. Roadmap additions: a **model-regression gate** (fail the PR if
LOSO κ drops > ε on a cached mini-dataset), **CML** to post the metrics table/confusion matrix
as a PR comment, and a tag-triggered job that builds + pushes the image and registers the model
in MLflow.

---

## 5. Phased execution plan

Each phase is independently shippable and leaves the repo in a working state. "DoD" = how you
know the phase is actually done.

**Phase 0 — Foundation (½–1 wk).** Adopt the package skeleton; move notebooks to
`notebooks/legacy/`; `pyproject` + `make install-dev`; pre-commit + CI green on an empty test
suite. *DoD: `pip install -e .` works; CI passes; no `.fif` is git-trackable.*

**Phase 1 — Port the classical pipeline (1–2 wk).** Implement loaders for your real datasets;
move filtering/epoching/cleaning into `preprocessing`; reproduce the `temporal analysis`
result through `evaluate.py` — but with **subject-aware CV** and the **Riemannian baseline**.
*DoD: one command reproduces a decoding number; leakage tests pass.*

**Phase 2 — Deep models + tracking (2–3 wk).** EEGNet → Conformer via Lightning; MLflow/W&B
logging; Optuna sweeps; nested CV. *DoD: EEGNet beats chance on LOSO and is logged; a sweep runs.*

**Phase 3 — Data versioning + reproducibility (1 wk).** DVC remote; `dvc repro` DAG; data hash
logged per run. *DoD: a teammate `git clone && dvc pull && dvc repro` reproduces your metric.*

**Phase 4 — A standout feature (2–4 wk).** Pick **one** of §3 to depth (real-time LSL *or* sleep
staging), plus the dashboard. *DoD: a live demo — stream→prediction, or staged hypnogram with
macro-F1.*

**Phase 5 — Serve + harden (1–2 wk).** Dockerised FastAPI + ONNX export; compose stack;
drift logging; model-regression gate in CI. *DoD: `docker compose up` serves `/predict`; PRs
are gated on metrics.*

**Phase 6 — Research edge (ongoing).** SSL pretraining / foundation-model fine-tuning; domain
adaptation; explainability in the dashboard.

---

## 6. Quality gates (definition of done, repo-wide)

- No plain `KFold`/`train_test_split` on multi-subject data — **only** the splitters in
  `data/splits.py`. (A grep for `KFold` outside `splits.py` should return nothing.)
- Stateful preprocessing fit **inside** the CV fold, never on the full set.
- Every reported number carries: dataset + DVC hash, resolved config, git SHA, seed, mean ± std
  across folds, and κ / balanced-acc / macro-F1 (not just accuracy).
- Raw signal is DVC-tracked, never committed; notebooks are output-stripped.
- New dataset = one `@register_loader` class; new model = one backbone + registry entry. No
  edits to training/serving code.

---

### Appendix — key dependencies by extra

`dl`: torch, lightning, braindecode, einops · `features`: pyriemann, PyWavelets, autoreject,
mne-icalabel · `mlops`: mlflow, wandb, dvc, optuna · `serve`: fastapi, uvicorn, pylsl,
streamlit, plotly. Install everything with `pip install -e ".[all]"`.

---

## 7. Phase 4 delivered — sleep-stage classification (implemented)

This phase is now built into the package, not just planned. It is the reference example of
the "sequence-of-epochs" task and reuses the same loaders, splits and tracking as everything
else — only the model shape and the metrics change.

### What was added

| Concern | File | What it does |
|---------|------|--------------|
| Dataset | `data/loaders.py::SleepEDFLoader` | Sleep-EDF → 30 s epochs, 5-class AASM labels, channel selection, wake-trimming. |
| Sequencing | `data/sequence.py` | `make_sequence_windows` (boundary-safe) + memory-light `SleepSequenceDataset` + `sequence_sampler_weights`. |
| Model | `models/sleep.py::TinySleepNet` | Per-epoch CNN encoder → BiLSTM → per-epoch head (many-to-many). |
| Training loop | `models/base.py::LitSequenceClassifier` | `(B,L,C,T)` loss, class-weighted **and** focal options, macro-F1/κ at epoch end. |
| Metrics | `utils/metrics.py::sleep_metrics` | accuracy, balanced-acc, macro-F1, κ, per-class F1. |
| Entry point | `training/sleep.py` (`eegpipe-sleep`) | Subject-aware split, class weights, weighted sampler, checkpoint on `val/macro_f1`. |
| Inference | `inference/hypnogram.py` | `HypnogramPredictor` (overlap-averaged) + `plot_hypnogram`. |
| Dashboard | `dashboard/app.py` | New **Hypnogram** tab. |
| Configs | `model/sleep_seqnet.yaml`, `training/sleep.yaml`, `preprocess/sleep.yaml`, `data/sleep_edf.yaml` | |

### The five design decisions that make it correct

1. **Many-to-many, not per-epoch.** A 30 s epoch is ambiguous alone; staging is a *sequence*
   problem. The model ingests L consecutive epochs and labels each, conditioned on neighbours —
   the single biggest accuracy lever (DeepSleepNet/SeqSleepNet/TinySleepNet all rely on it).
2. **Sequences never cross a night.** `make_sequence_windows` only builds windows inside a
   contiguous recording, so context never leaks across subjects/nights — the sequence analogue
   of subject-aware CV.
3. **Imbalance is handled three ways at once.** Inverse-frequency **class weights** (computed on
   train epochs only), an optional **WeightedRandomSampler** that oversamples sequences
   containing rare stages (N1), and **focal loss** as a fallback. Selection metric is
   **macro-F1**, never accuracy — an all-N2 predictor scores ~50% accuracy and is worthless.
4. **Right preprocessing.** `preprocess=sleep` band-passes 0.3–35 Hz (the default 8–32 Hz MI
   filter would delete delta/slow-wave activity — the exact signal N3 staging depends on) and
   resamples to 100 Hz to match the encoder's filter sizing.
5. **Robust full-night inference.** `HypnogramPredictor` slides dense, end-padded windows and
   **averages the softmax** over every window covering an epoch, smoothing boundary effects.

### Run it

```bash
eegpipe-sleep data=sleep_edf model=sleep_seqnet training=sleep preprocess=sleep \
    data.n_subjects=20 training.logger=mlflow
eegpipe-sleep ... training.fast_dev_run=true        # 1-batch smoke test
```

The run logs per-class F1 (watch **N1** — it's the hardest), κ and macro-F1, writes
`reports/sleep_metrics.json`, and checkpoints the best macro-F1 model to `models_store/`.
Visualise a night: `make dashboard` → **Hypnogram** tab → point at the checkpoint.

### Where to take it next (Phase 4+)

- **Stronger sequence backbone**: swap the BiLSTM for a Transformer encoder, or adopt a
  fully-convolutional **U-Time** (segmentation-style) head for arbitrary-length nights.
- **Attention over epochs** to expose *which* neighbouring epochs drove a stage call
  (clinician-facing explainability in the dashboard).
- **Multi-modal**: add EOG + EMG channels (the loader already supports a channel list) — chin
  EMG is decisive for REM, EOG for Wake/REM.
- **Transfer**: pretrain the epoch encoder with the SSL relative-positioning task
  (`models/ssl.py`) on unlabelled nights, then fine-tune — directly attacks the small-labelled-
  data regime typical of new clinical sites.

---

## 8. Phase 4+ / Phase 5 delivered — stronger models, SSL, and production serving

### 8.1 Stronger sleep backbones (a)

Two backbones now sit alongside `TinySleepNet`, all three sharing one `EpochEncoder` (so the
SSL-pretrained encoder in §8.2 drops into any of them):

| Backbone | File | Context model | Reach for it when |
|----------|------|---------------|-------------------|
| `sleep_seqnet` | `models/sleep.py::TinySleepNet` | BiLSTM | Fast, strong default. |
| `sleep_transformer` | `…::SleepTransformer` | Self-attention over epochs (+ positional encoding) | More data; long-range dependencies (sleep-cycle structure). |
| `utime` | `…::UTime` | Multi-scale temporal U-Net over epoch embeddings | Want both fine transitions (N1↔N2) and night-scale context; arbitrary-length nights (auto-padded). |

```bash
eegpipe-sleep data=sleep_edf model=sleep_transformer training=sleep preprocess=sleep
eegpipe-sleep data=sleep_edf model=utime           training=sleep preprocess=sleep
```

**Multimodal (EOG/EMG) — mainly helps REM.** `data=sleep_edf_multimodal` adds `EOG horizontal`
and `EMG submental`. REM is defined by EEG + **REM (EOG)** + **chin-EMG atonia**; an EEG-only
model routinely confuses REM with N1/Wake. The models read `n_channels` from the data, so no
code change is needed — only the config:

```bash
eegpipe-sleep data=sleep_edf_multimodal model=sleep_transformer training=sleep preprocess=sleep
```

(MNE upsamples the lower-rate EMG channel on read; `preprocess=sleep` then resamples to 100 Hz.)

### 8.2 Self-supervised pretraining for the low-label regime (b)

`models/ssl.py` implements **Relative Positioning**: sample two epochs from the same night,
label = "are they temporally close?". The encoder learns from *unlabelled* nights; you then
fine-tune on the few labelled ones. The pretext encoder **is** the sleep models' `EpochEncoder`,
so transfer is a weight-load, not a re-architecture.

```bash
# 1) Pretrain the encoder on unlabelled recordings
eegpipe-pretrain data=sleep_edf model=sleep_seqnet training=pretrain preprocess=sleep
# 2) Fine-tune the full sleep model, warm-starting the encoder
eegpipe-sleep data=sleep_edf model=sleep_seqnet training=sleep preprocess=sleep \
    model.pretrained_encoder=models_store/ssl_encoder.pt
```

`models.registry.load_pretrained_encoder` loads with `strict=False`, so a partial match (e.g.
different channel count) degrades gracefully rather than crashing. Next: contrastive SSL and
masked autoencoding reuse the same encoder slot; or fine-tune a public foundation encoder
(BENDR/LaBraM) by adapting it to the `EpochEncoder` interface.

### 8.3 Phase 5 — export, serving hardening, regression gate (c)

- **Export** (`serving/export.py`, `eegpipe-export`): ONNX **and** TorchScript, with batch (and
  sequence length) as dynamic axes. `export_module_onnx` is checkpoint-independent so it's unit-
  tested without training.
  ```bash
  eegpipe-export model.ckpt model.onnx --kind trial --channels 22 --times 256
  eegpipe-export sleep.ckpt sleep.onnx --kind sequence --channels 4 --times 3000 --seq-len 20
  ```
- **ONNX-first serving** (`inference/onnx_predictor.py`): `OnnxPredictor` mirrors `Predictor`'s
  interface but runs on ONNX Runtime (2–5× faster on CPU, no torch in the serving image).
- **Hardened FastAPI** (`serving/api.py`): backend auto-selected (`EEGPIPE_ONNX` ▸ `EEGPIPE_CKPT`),
  `/health` (liveness) vs `/ready` (readiness) vs `/version`, request-timing header, optional
  API-key auth (`EEGPIPE_API_KEY`), input-size guard (413), and model name/version/backend +
  latency in every response.
- **Model-regression gate** (`scripts/check_regression.py`, `make regression`, CI job
  `regression-gate`): fails the build if a key metric drops more than `--tol` below the committed
  `tests/baseline_metrics.json`. Wire it to the latest run's `reports/sleep_metrics.json`; promote
  a new baseline only via a reviewed commit. This is the safety net that stops a refactor or data
  change from silently degrading the model.

### 8.4 What to watch

- **N1 F1** remains the canary — it's the rarest, most confusable stage. If class weights +
  weighted sampler don't lift it, try `model.use_focal=true`, then multimodal (EOG/EMG), then SSL.
- **macro-F1 / κ**, never accuracy, gate both model selection (early-stopping monitor) and CI.

---

## 9. Trust, transfer & deployment (delivered)

### 9.1 Interpretability — `interpret/` (clinician trust)

A model that can't explain itself won't be trusted in a clinic. Three artefacts:

- **Integrated-Gradients saliency** (`interpret/saliency.py`) — attributes a per-trial
  prediction to channels × time-points; `channel_importance` collapses it to a per-electrode
  score. No Captum dependency.
- **Attention rollout** (`interpret/attention.py`) — for `SleepTransformer`, the *real*
  per-layer epoch×epoch attention (we re-run the layers transparently rather than fight the
  fused fast path), composed via Abnar–Zuidema rollout into one map: "which neighbouring epochs
  drove this stage call?".
- **Confusion matrix** — `LitSequenceClassifier` now stores the test confusion matrix;
  `eegpipe-sleep` writes `reports/confusion.npy`, and the dashboard's new **Explain** tab renders
  it alongside a live saliency map. This is where you *see* that N1 is the model's weak spot.

### 9.2 Cross-site domain adaptation — `adaptation/`

EEG distributions shift across subjects, sessions and headsets, so an off-site model degrades.
Both methods are **unsupervised**, so a brand-new test subject is aligned by its own statistics
— no labels, no leakage:

- **Euclidean Alignment** (`EuclideanAlignment`, He & Wu 2020) — whiten trials by the domain's
  `R^{-1/2}`; afterwards every domain's mean covariance is the identity (unit-tested to ~1e-4).
  Enable in the classical CV with `align=true`.
- **Riemannian Recentering** (`RiemannianRecentering`, Zanini 2018) — recentre covariances to
  the identity on the SPD manifold; pairs with the Riemannian tangent-space classifier.

```bash
eegpipe-eval data=bci_iv_2a loso=true align=true   # LOSO with Euclidean Alignment
```

### 9.3 Foundation-model encoder — `models/foundation.py`

`FoundationEncoderAdapter` normalises any pretrained EEG backbone (**BENDR**, **LaBraM**, …) to
the `EpochEncoder` contract, with a 1×1 **channel adapter** (montage → backbone channels) and a
projection to `emb_dim`. Because the sleep backbones now take an injectable `encoder=`, a
foundation model drops in via config:

```bash
eegpipe-sleep data=sleep_edf model=sleep_foundation training=sleep preprocess=sleep
#   model=sleep_foundation -> encoder=foundation, foundation=mock (runs anywhere).
#   Swap foundation=bendr|labram after installing the upstream repo + weights.
```

`foundation=mock` is a runnable stand-in so the whole transfer path is tested in CI; real
weights are separately licensed (the loader raises with install instructions). Encoder selection
is unified: `encoder=foundation` (foundation model) vs `pretrained_encoder=...` (SSL weights, §8.2)
vs neither (fresh encoder).

### 9.4 Deployment — `deploy/`

Manifests + a decision guide (`deploy/README.md`), all serving the exported **ONNX** artefact:

- **`deploy/k8s/`** — Deployment with `/ready` (readiness) + `/health` (liveness) probes,
  Service, and an HPA (CPU 70%, 2→10 pods).
- **`deploy/kserve/`** — KServe `InferenceService` (Knative autoscaling incl. scale-to-zero,
  canary splits); custom-container *or* managed-Triton variants.
- **`deploy/triton/`** — Triton `config.pbtxt` with server-side **dynamic batching** for max
  throughput; one line to flip CPU→GPU.

Rule kept front-and-centre: don't reach for k8s until one container is actually the bottleneck.
Promotions are gated by the CI regression gate (§8.3); drift on input-feature distributions is
expected for EEG and should be monitored.

---

*Status: the project now spans the full lifecycle — data/BIDS → leakage-safe CV → classical
(Riemannian) and deep (EEGNet/Conformer/ST-GCN) decoding → sleep-staging sequence models
(LSTM/Transformer/U-Time) with SSL pretraining and foundation-model transfer → interpretability
and domain adaptation → ONNX export, a hardened API, and Triton/KServe/k8s deployment, all under
Hydra configs, DVC, experiment tracking, tests and CI.*

---

## 10. End-to-end smoke test (`make demo`)

One command that walks the **entire pipeline** and reports `PASS / SKIP / FAIL` per stage:
data → sequence windows → subject-aware split → Euclidean alignment → classical band-power
baseline → deep `fast_dev_run` → ONNX export + runtime parity → hypnogram → regression gate.

```bash
make demo                 # synthetic, offline (classical stages run; deep stages need torch)
make demo REAL=--real     # one real Sleep-EDF subject (downloads via MNE)
python scripts/smoke_demo.py --quick   # tiny synthetic run (used by tests/test_smoke_demo.py)
```

Stages that need an optional dep (torch/onnx) **SKIP** rather than fail, so the demo runs
anywhere and tells you exactly what your environment can and can't do.

**Why this matters (a real find).** Running the `--real` path surfaced a bug that `py_compile`
and import-smoke had *missed*: a formatter had silently truncated `SleepEDFLoader.load`, leaving
the class abstract. The file still compiled and imported — the failure only appeared at
*instantiation*. The lesson, now baked into the repo: unit tests and import checks are necessary
but not sufficient; an end-to-end smoke that actually constructs and runs each component is what
catches this class of regression. A companion AST integrity check verifies every expected
class/method is present across all modules, guarding against truncation going forward.
