"""Whisper transcription regression tests.

The pipeline has already decoded and normalized speech with soundfile. These
tests pin the handoff to faster-whisper as a NumPy waveform so transcription
cannot regress to faster-whisper's file-decoding (PyAV) path.
"""

from types import SimpleNamespace

import numpy as np

from mci_biomarker.config import WhisperConfig
from mci_biomarker.linguistic import transcribe as transcribe_module


class FakeModel:
    def __init__(self, segments):
        self._segments = segments
        self.audio = None
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return iter(self._segments), SimpleNamespace(language="en")


def segment(text="hello world", avg_logprob=-0.2):
    return SimpleNamespace(start=0.0, end=1.0, text=text, avg_logprob=avg_logprob)


def test_transcribe_passes_waveform_not_file_path(monkeypatch):
    model = FakeModel([segment()])
    monkeypatch.setattr(transcribe_module, "_load_whisper_model", lambda model_name, cfg: model)
    speech = np.linspace(-0.1, 0.1, 16000, dtype=np.float64)

    text, segments, confidence, backend = transcribe_module.transcribe(
        speech, 16000, WhisperConfig()
    )

    assert isinstance(model.audio, np.ndarray)
    assert model.audio.dtype == np.float32
    assert model.audio.shape == (16000,)
    assert text == "hello world"
    assert len(segments) == 1
    assert confidence is not None and confidence > 0
    assert backend == "faster-whisper"


def test_transcribe_resamples_waveform_to_whisper_sample_rate(monkeypatch):
    model = FakeModel([segment()])
    monkeypatch.setattr(transcribe_module, "_load_whisper_model", lambda model_name, cfg: model)
    speech = np.zeros(8000, dtype=np.float32)

    transcribe_module.transcribe(speech, 8000, WhisperConfig())

    assert isinstance(model.audio, np.ndarray)
    assert abs(model.audio.shape[0] - 16000) <= 2


def test_transcribe_failure_returns_empty_transcript_with_warning(monkeypatch):
    def failing_segments():
        yield segment("partial result that should be discarded")
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    model = FakeModel(failing_segments())
    monkeypatch.setattr(transcribe_module, "_load_whisper_model", lambda model_name, cfg: model)
    warnings: list[str] = []

    text, segments, confidence, backend = transcribe_module.transcribe(
        np.zeros(16000, dtype=np.float32),
        16000,
        WhisperConfig(),
        warnings=warnings,
    )

    assert text == ""
    assert segments == []
    assert confidence is None
    assert backend == "none"
    assert any("metadata_errors" in warning for warning in warnings)
