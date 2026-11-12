"""Cantonese mode (--language yue): segmentation, metrics, ASR routing, wiring."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from mci_biomarker.baseline.store import ProfileStore, compare_to_baseline
from mci_biomarker.config import PipelineConfig, WhisperConfig
from mci_biomarker.linguistic import cantonese
from mci_biomarker.linguistic import transcribe as transcribe_module
from mci_biomarker.linguistic.features import extract_linguistic, tokenize
from mci_biomarker.pipeline import analyze_wav

SR = 16000
RICH_YUE = (
    "個媽媽喺廚房度洗碗，水喉啲水流到成地都係。"
    "個男仔企喺張凳上面攞個罐入面啲曲奇，佢妹妹喺隔籬望住佢。"
    "窗邊有啲杯同埋碟放喺枱上面。"
)
VAGUE_YUE = "嗯，佢將嗰樣嘢俾咗我，即係嗰個，嗰樣嘢，嗯，佢又攞嗰樣嘢，嗰個嘢，嗯即係嗰樣嘢啦。"


def synth_wav(path: Path) -> Path:
    t = np.arange(int(2.4 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t))
    signal = np.concatenate([tone[: SR], np.zeros(SR // 2), tone[:SR]]).astype(np.float32)
    sf.write(str(path), signal, SR)
    return path


@pytest.fixture()
def cfg(tmp_path):
    c = PipelineConfig()
    c.diarization = "energy"
    c.acoustic_backend = "signal"
    c.baseline.profile_dir = tmp_path / "profiles"
    c.baseline.min_baseline_sessions = 2
    return c


# --- Segmentation + linguistic metrics -----------------------------------------

def test_maxmatch_segmentation_finds_cantonese_words():
    tokens = cantonese._maxmatch_segment(RICH_YUE)
    for word in ("媽媽", "廚房", "洗碗", "水喉", "男仔", "曲奇", "妹妹"):
        assert word in tokens, word
    assert len(tokens) < len(RICH_YUE)  # words, not raw characters


def test_tokenize_keeps_english_codeswitching():
    tokens = tokenize("我今日去咗school返學", "yue")
    assert "school" in tokens


def test_simplified_asr_output_is_normalized():
    # Qwen3-ASR writes simplified chars (个/系/时); metrics must still work.
    simp = "佢将嗰样嘢俾咗我，嗰个系我当时买嘅。嗯，即系嗰个嘢啦。"
    feats = extract_linguistic(simp, backend="qwen3-asr", language="yue")
    assert feats.pronoun_count >= 3  # 佢 / 我×2 (+嗰個 after normalization)
    assert feats.vague_word_count >= 1  # 嗰樣嘢/個嘢 after normalization
    assert feats.filler_count >= 2  # 嗯 + 即系
    assert cantonese.normalize_traditional("嗰个系") == "嗰個係"


def test_cantonese_metrics_flag_vague_speech():
    rich = extract_linguistic(RICH_YUE, backend="manual", language="yue")
    vague = extract_linguistic(VAGUE_YUE, backend="manual", language="yue")
    assert rich.language == vague.language == "yue"
    assert rich.ttr is not None and 0 < rich.ttr <= 1
    assert vague.pronoun_count >= 3  # 佢/我/嗰個...
    assert vague.vague_word_count >= 3  # 嗰樣嘢 ×3 + 嘢
    assert vague.filler_count >= 3  # 嗯 / 即係
    assert vague.pronoun_to_noun_ratio is not None
    assert rich.pronoun_to_noun_ratio is not None
    assert vague.pronoun_to_noun_ratio > rich.pronoun_to_noun_ratio


# --- ASR backend routing ---------------------------------------------------------

class FakeWhisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        seg = SimpleNamespace(start=0.0, end=1.0, text="你好嗎", avg_logprob=-0.3)
        return iter([seg]), SimpleNamespace(language="yue")


def test_yue_prefers_qwen3_backend(monkeypatch):
    calls = []

    def fake_qwen(waveform, cfg, lang, duration_s):
        calls.append(("qwen3", lang))
        return "你好，今日天氣好好。", []

    monkeypatch.setattr(transcribe_module, "_transcribe_qwen3", fake_qwen)
    monkeypatch.setattr(
        transcribe_module, "_transcribe_sensevoice",
        lambda *a: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    text, _segs, conf, backend = transcribe_module.transcribe(
        np.zeros(SR, dtype=np.float32), SR, WhisperConfig(), language="yue"
    )
    assert backend == "qwen3-asr"
    assert text.startswith("你好")
    assert conf is None
    assert calls == [("qwen3", "yue")]


def test_yue_falls_back_through_chain_to_whisper_large_v3(monkeypatch):
    seen = {}

    def boom(*args):
        raise RuntimeError("not installed")

    fake = FakeWhisper()

    def fake_load(model_name, cfg):
        seen["model"] = model_name
        return fake

    monkeypatch.setattr(transcribe_module, "_transcribe_qwen3", boom)
    monkeypatch.setattr(transcribe_module, "_transcribe_sensevoice", boom)
    monkeypatch.setattr(transcribe_module, "_load_whisper_model", fake_load)
    warnings: list[str] = []
    cfg = WhisperConfig()  # default model tiny.en (English-only)
    text, _segs, _conf, backend = transcribe_module.transcribe(
        np.zeros(SR, dtype=np.float32), SR, cfg, language="yue", warnings=warnings
    )
    assert backend == "faster-whisper"
    assert seen["model"] == "large-v3-turbo"  # auto-switched away from *.en
    assert fake.kwargs["language"] == "yue"
    assert any("English-only" in w for w in warnings)


def test_english_stays_on_whisper_only(monkeypatch):
    monkeypatch.setattr(
        transcribe_module, "_transcribe_qwen3",
        lambda *a: (_ for _ in ()).throw(AssertionError("qwen3 must not run for en")),
    )
    fake = FakeWhisper()
    monkeypatch.setattr(transcribe_module, "_load_whisper_model", lambda m, c: fake)
    _text, _segs, _conf, backend = transcribe_module.transcribe(
        np.zeros(SR, dtype=np.float32), SR, WhisperConfig(), language="en"
    )
    assert backend == "faster-whisper"
    assert fake.kwargs["language"] == "en"


# --- Pipeline / baseline / web wiring --------------------------------------------

def test_pipeline_cantonese_mode(tmp_path, cfg):
    wav = synth_wav(tmp_path / "yue.wav")
    profile = analyze_wav(wav, user_id="yue-u", cfg=cfg,
                           transcript_override=RICH_YUE, language="yue")
    assert profile.linguistic.language == "yue"
    assert profile.linguistic.pronoun_to_noun_ratio is not None
    assert profile.linguistic.token_count > 10


def test_baseline_never_mixes_languages(tmp_path, cfg):
    wav = synth_wav(tmp_path / "s.wav")
    en = "The mother washes dishes while the boy takes cookies from the jar."
    store = ProfileStore(cfg.baseline.profile_dir)
    for _ in range(3):
        store.save_profile(analyze_wav(wav, user_id="mix", cfg=cfg,
                                       transcript_override=en, language="en"))
    yue_profile = analyze_wav(wav, user_id="mix", cfg=cfg,
                              transcript_override=RICH_YUE, language="yue")
    comp = compare_to_baseline(yue_profile, store.load_profiles("mix"), min_sessions=2)
    assert comp.n_prior_sessions == 0  # English priors excluded
    assert comp.baseline_ready is False


def test_web_language_field(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    monkeypatch.setattr(web_app, "_store", lambda: ProfileStore(cfg.baseline.profile_dir))
    client = TestClient(web_app.app)
    wav = synth_wav(tmp_path / "up.wav")
    with wav.open("rb") as fh:
        resp = client.post(
            "/api/analyze",
            files={"file": ("up.wav", fh, "audio/wav")},
            data={"user_id": "yue-web", "transcript": RICH_YUE,
                  "language": "yue", "save": "false"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["linguistic"]["language"] == "yue"
    assert body["linguistic"]["pronoun_to_noun_ratio"] is not None
