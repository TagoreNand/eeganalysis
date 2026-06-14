"""Concrete dataset strategies.

Each loader normalises a very different on-disk format into one :class:`EpochsBundle`.
All are registered with :data:`LOADER_REGISTRY` so ``build_loader("bci_iv_2a")`` works.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eegpipe.data.base import BaseDataLoader, EpochsBundle
from eegpipe.data.registry import register_loader
from eegpipe.utils import get_logger

log = get_logger(__name__)


@register_loader("mne_sample")
class MNESampleLoader(BaseDataLoader):
    """The audvis ERP set used in the original notebooks — Left/Right auditory decoding.

    Single subject, so use this for within-subject decoding and fast smoke tests only;
    LOSO is meaningless here (one group).
    """

    name = "mne_sample"

    def __init__(self, l_freq: float = 0.1, h_freq: float = 40.0, tmin=-0.2, tmax=0.5, **kw):
        super().__init__(**kw)
        self.l_freq, self.h_freq, self.tmin, self.tmax = l_freq, h_freq, tmin, tmax

    def load(self, subjects: Sequence[int | str] | None = None) -> EpochsBundle:
        import mne

        data_dir = mne.datasets.sample.data_path()
        raw = mne.io.read_raw_fif(f"{data_dir}/MEG/sample/sample_audvis_raw.fif", preload=True)
        raw.pick(["eeg", "eog"]).filter(self.l_freq, self.h_freq, verbose="error")
        events = mne.find_events(raw, stim_channel="STI 014", verbose="error")
        event_id = {"Auditory/Left": 1, "Auditory/Right": 2}
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            self.tmin,
            self.tmax,
            baseline=(None, 0),
            preload=True,
            verbose="error",
        )
        epochs.equalize_event_counts(list(event_id))
        subject_ids = np.zeros(len(epochs), dtype=int)  # single subject
        return EpochsBundle(
            epochs,
            subject_ids,
            "auditory_side",
            metadata={"dataset": "mne_sample", "n_subjects": 1},
        )


@register_loader("bci_iv_2a")
class MOABBLoader(BaseDataLoader):
    """Cross-subject motor-imagery via MOABB (default: BCI Competition IV-2a, 9 subjects).

    MOABB handles download, BIDS-like caching, and crucially exposes a ``subject`` column
    that we map straight onto ``subject_ids`` — enabling real LOSO evaluation.
    """

    name = "bci_iv_2a"

    def __init__(
        self,
        dataset: str = "BNCI2014_001",
        paradigm: str = "MotorImagery",
        fmin: float = 8.0,
        fmax: float = 32.0,
        **kw,
    ):
        super().__init__(**kw)
        self.dataset, self.paradigm = dataset, paradigm
        self.fmin, self.fmax = fmin, fmax

    def load(self, subjects: Sequence[int | str] | None = None) -> EpochsBundle:
        import moabb.datasets as mds
        from moabb.paradigms import MotorImagery

        ds = getattr(mds, self.dataset)()
        if subjects is not None:
            ds.subject_list = list(subjects)
        paradigm = MotorImagery(fmin=self.fmin, fmax=self.fmax)
        # return_epochs=True keeps an MNE object so our preprocessing transforms still apply
        epochs, labels, meta = paradigm.get_data(ds, return_epochs=True)
        # Encode string labels -> ints, store mapping in metadata
        classes, y = np.unique(labels, return_inverse=True)
        epochs.events[:, 2] = y
        subject_ids = meta["subject"].to_numpy()
        return EpochsBundle(
            epochs,
            subject_ids,
            "motor_imagery",
            metadata={
                "dataset": self.dataset,
                "classes": classes.tolist(),
                "n_subjects": int(meta["subject"].nunique()),
            },
        )


@register_loader("bids")
class BIDSDataLoader(BaseDataLoader):
    """Read an EEG-BIDS dataset. Pairs with the original 'understanding with bids' notebook."""

    name = "bids"

    def __init__(
        self, root: str, task: str, suffix: str = "eeg", tmin: float = -0.2, tmax: float = 0.8, **kw
    ):
        super().__init__(**kw)
        self.root, self.task, self.suffix = root, task, suffix
        self.tmin, self.tmax = tmin, tmax

    def load(self, subjects: Sequence[int | str] | None = None) -> EpochsBundle:
        import mne
        from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

        subs = list(subjects) if subjects else get_entity_vals(self.root, "subject")
        all_epochs, groups = [], []
        for sub in subs:
            bp = BIDSPath(
                subject=str(sub), task=self.task, suffix=self.suffix, datatype="eeg", root=self.root
            )
            raw = read_raw_bids(bp, verbose="error").load_data()
            events, event_id = mne.events_from_annotations(raw, verbose="error")
            ep = mne.Epochs(
                raw,
                events,
                event_id,
                self.tmin,
                self.tmax,
                baseline=None,
                preload=True,
                verbose="error",
            )
            all_epochs.append(ep)
            groups.append(np.full(len(ep), int(sub) if str(sub).isdigit() else hash(sub) % 10_000))
        epochs = mne.concatenate_epochs(all_epochs)
        return EpochsBundle(
            epochs,
            np.concatenate(groups),
            "bids_task",
            metadata={"dataset": f"bids:{self.root}", "task": self.task},
        )


@register_loader("sleep_edf")
class SleepEDFLoader(BaseDataLoader):
    """Sleep-EDF (PhysioNet) -> 30s epochs labelled with AASM sleep stages.

    Stages are highly imbalanced (N2 dominates), so train with stratified splits and a
    class-weighted loss (see ``models/base.py``). Channels default to the two EEG
    derivations; add EOG/EMG via ``channels`` for multimodal staging.
    """

    name = "sleep_edf"
    STAGE_MAP = {  # AASM 5-class (N3/N4 merged per current guidelines)
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

    def __init__(
        self,
        n_subjects=20,
        channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
        recording=(1,),
        crop_wake_mins=30.0,
        **kw,
    ):
        super().__init__(**kw)
        self.n_subjects = n_subjects
        self.channels = list(channels)
        self.recording = list(recording)
        self.crop_wake_mins = crop_wake_mins

    def _crop_wake(self, raw):
        """Trim long Wake stretches before/after the sleep period (standard for Sleep-EDF)."""
        if not self.crop_wake_mins:
            return
        ann = raw.annotations
        onsets = [
            o
            for o, d in zip(ann.onset, ann.description, strict=False)
            if d in self.STAGE_MAP and d != "Sleep stage W"
        ]
        if not onsets:
            return
        pad = self.crop_wake_mins * 60.0
        tmin = max(float(raw.times[0]), min(onsets) - pad)
        tmax = min(float(raw.times[-1]), max(onsets) + pad)
        raw.crop(tmin=tmin, tmax=tmax)

    def load(self, subjects=None):
        import mne
        from mne.datasets.sleep_physionet.age import fetch_data

        subs = list(subjects) if subjects else list(range(self.n_subjects))
        all_ep, groups = [], []
        for sub in subs:
            psg, hyp = fetch_data([sub], recording=self.recording, verbose="error")[0]
            raw = mne.io.read_raw_edf(psg, preload=True, verbose="error")
            raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
            present = [c for c in self.channels if c in raw.ch_names]
            raw.pick(present or "eeg")
            self._crop_wake(raw)
            events, _ = mne.events_from_annotations(
                raw, event_id=self.STAGE_MAP, chunk_duration=30.0, verbose="error"
            )
            ep = mne.Epochs(
                raw,
                events,
                tmin=0.0,
                tmax=30.0 - 1.0 / raw.info["sfreq"],
                baseline=None,
                preload=True,
                verbose="error",
            )
            all_ep.append(ep)
            groups.append(np.full(len(ep), sub))
        epochs = mne.concatenate_epochs(all_ep)
        return EpochsBundle(
            epochs,
            np.concatenate(groups),
            "sleep_stage",
            metadata={
                "dataset": "sleep_edf",
                "n_classes": 5,
                "stage_names": self.STAGE_NAMES,
                "n_subjects": len(subs),
            },
        )
