"""Streamlit dashboard: load EEG, inspect signals, and visualise model predictions.

Run: ``streamlit run src/eegpipe/dashboard/app.py`` (or ``make dashboard``).

Tabs:
  1. Signals    — raw traces + PSD
  2. Topomap    — scalp distribution for a chosen frequency band
  3. Predict    — per-trial checkpoint -> per-class probabilities
  4. Hypnogram  — sequence model -> full-night sleep stages
  5. Explain    — confusion matrix + Integrated-Gradients saliency (clinician trust)
"""

from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="eegpipe", layout="wide")
st.title("eegpipe — EEG analysis & inference dashboard")


@st.cache_data(show_spinner="Loading sample data...")
def _load_sample():
    import mne

    d = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(f"{d}/MEG/sample/sample_audvis_raw.fif", preload=True)
    return raw.pick("eeg").filter(1, 40, verbose="error")


source = st.sidebar.radio("Data source", ["MNE sample", "Upload .fif / .edf"])
raw = None
if source == "MNE sample":
    raw = _load_sample()
else:
    up = st.sidebar.file_uploader("EEG file", type=["fif", "edf", "bdf"])
    if up:
        import tempfile

        import mne

        with tempfile.NamedTemporaryFile(delete=False, suffix=up.name[-4:]) as f:
            f.write(up.getbuffer())
            path = f.name
        reader = mne.io.read_raw_fif if up.name.endswith("fif") else mne.io.read_raw_edf
        raw = reader(path, preload=True)

if raw is None:
    st.info("Choose or upload a recording to begin.")
    st.stop()

tabs = st.tabs(["Signals", "Topomap", "Predict", "Hypnogram", "Explain"])
tab_sig, tab_topo, tab_pred, tab_hyp, tab_explain = tabs

with tab_sig:
    st.subheader("Raw traces & power spectral density")
    c1, c2 = st.columns(2)
    dur = st.slider("Window (s)", 1, 20, 5)
    sfreq = raw.info["sfreq"]
    data, _ = raw[:8, : int(dur * sfreq)]
    c1.line_chart({raw.ch_names[i]: data[i] for i in range(min(4, data.shape[0]))})
    psd = raw.compute_psd(fmax=45, verbose="error")
    c2.line_chart({"PSD (dB)": 10 * np.log10(psd.get_data().mean(0) + 1e-20)})

with tab_topo:
    st.subheader("Scalp topography")
    band = st.selectbox("Band", ["delta", "theta", "alpha", "beta", "gamma"], index=2)
    import matplotlib.pyplot as plt

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    lo, hi = bands[band]
    psd = raw.compute_psd(fmin=lo, fmax=hi, verbose="error")
    fig, ax = plt.subplots()
    try:
        import mne

        mne.viz.plot_topomap(psd.get_data().mean(1), raw.info, axes=ax, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Topomap needs channel positions: {e}")

with tab_pred:
    st.subheader("Model predictions (per-trial)")
    ckpt = st.text_input("Checkpoint path (.ckpt)", value="")
    if ckpt:
        try:
            from eegpipe.inference import Predictor

            predictor = Predictor(ckpt)
            X = raw.get_data()[None, :, :512].astype("float32")
            proba = predictor.predict_proba(X)[0]
            st.bar_chart({f"class_{i}": float(p) for i, p in enumerate(proba)})
        except Exception as e:
            st.error(f"Could not run model: {e}")
    else:
        st.caption("Provide a trained checkpoint to see live predictions.")

with tab_hyp:
    st.subheader("Sleep hypnogram (sequence model)")
    st.caption(
        "Score this recording into AASM stages with a checkpoint trained via "
        "eegpipe-sleep. Resamples to 100 Hz to match training."
    )
    ck = st.text_input("Sequence checkpoint (.ckpt)", value="", key="hyp_ckpt")
    seq_len = st.slider("Sequence length (epochs)", 5, 40, 20)
    if ck:
        try:
            import mne

            from eegpipe.inference import HypnogramPredictor, plot_hypnogram

            r = raw.copy().resample(100, verbose="error")
            epo = mne.make_fixed_length_epochs(r, duration=30.0, preload=True, verbose="error")
            X = epo.get_data(copy=False).astype("float32")
            predictor = HypnogramPredictor(ck, seq_len=seq_len)
            stages, _ = predictor.predict(X)
            ax = plot_hypnogram(stages, stage_names=predictor.stage_names)
            st.pyplot(ax.figure)
            names = predictor.stage_names
            st.bar_chart({names[i]: int((stages == i).sum()) for i in range(len(names))})
        except Exception as e:
            st.error(f"Could not produce hypnogram: {e}")
    else:
        st.caption("Provide a trained sequence checkpoint to see the predicted hypnogram.")

with tab_explain:
    st.subheader("Explainability")
    st.caption(
        "Evidence behind the model: where it confuses stages, and which "
        "channels/time-points drive a single prediction."
    )

    st.markdown("**Confusion matrix** (from the last `eegpipe-sleep` test run)")
    import json
    import os

    if os.path.exists("reports/confusion.npy"):
        from eegpipe.interpret import plot_confusion_matrix

        cm = np.load("reports/confusion.npy")
        if os.path.exists("reports/confusion_labels.json"):
            with open("reports/confusion_labels.json") as f:
                labels = json.load(f)
        else:
            labels = [str(i) for i in range(len(cm))]
        st.pyplot(plot_confusion_matrix(cm, [str(x) for x in labels]).figure)
    else:
        st.caption("Run `eegpipe-sleep` to produce reports/confusion.npy.")

    st.markdown("**Integrated-Gradients saliency** for a per-trial checkpoint")
    sc = st.text_input("Per-trial checkpoint (.ckpt)", value="", key="sal_ckpt")
    if sc:
        try:
            from eegpipe.interpret import integrated_gradients, plot_saliency
            from eegpipe.models.base import LitEEGClassifier

            model = LitEEGClassifier.load_from_checkpoint(sc, map_location="cpu").eval()
            X = raw.get_data()[:, :512].astype("float32")
            attr = integrated_gradients(model, X)
            st.pyplot(plot_saliency(attr, ch_names=raw.ch_names).figure)
        except Exception as e:
            st.error(f"Could not compute saliency: {e}")
    else:
        st.caption("Provide a per-trial checkpoint to see channel/time attributions.")
