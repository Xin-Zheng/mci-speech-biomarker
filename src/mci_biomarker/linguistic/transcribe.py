"""ASR transcription with language-aware backend routing (SPEC Track B).

Backends (all local / open-source, tried in order under `asr_backend="auto"`):

- Cantonese (`yue`): Qwen3-ASR → SenseVoice (FunASR) → faster-whisper with the
  `yue` language token (large-v3+ only). Qwen3-ASR is preferred because it
  preserves spoken Cantonese characters (佢/嘅/嘢) that our pronoun/vague-word
  biomarkers depend on, where Whisper often rewrites them into standard
  written Chinese.
- Other languages: faster-whisper (as before).

Confidence = mean exp(avg_logprob) across Whisper segments; Qwen3/SenseVoice
do not expose comparable scores, so confidence is None there. When Whisper
confidence falls below threshold the pipeline flags linguistic metrics
low-confidence and leans on acoustics (SPEC §4 dialect robustness).

Every backend fails soft: an unavailable/broken engine yields an empty
transcript + warning so the acoustic report still succeeds.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np

from ..audio.ingest import TARGET_SR, resample_to, write_wav
from ..config import WhisperConfig
from ..schemas import TranscriptSegment
from .cantonese import is_cjk

_model_cache: dict[str, object] = {}

# Languages routed through the Cantonese-first chain (SenseVoice covers zh too).
_CHAIN_LANGUAGES = {"yue", "cantonese", "zh-hk"}
_QWEN_LANGUAGE_NAMES = {
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "zh-hk": "Cantonese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
}


def transcribe(
    speech: np.ndarray,
    sr: int,
    cfg: WhisperConfig,
    language: str | None = None,
    warnings: list[str] | None = None,
    segment_cb=None,
) -> tuple[str, list[TranscriptSegment], float | None, str]:
    """Return (transcript, segments, mean_confidence, backend)."""
    warnings = warnings if warnings is not None else []
    if speech.size == 0:
        warnings.append("No speech detected; skipped transcription.")
        return "", [], None, "none"
    waveform = _prepare_waveform(speech, sr)
    duration_s = float(waveform.size) / TARGET_SR
    lang = (language or "en").lower()

    for backend in _backend_chain(cfg.asr_backend, lang):
        try:
            if backend == "qwen3":
                text, segments = _transcribe_qwen3(waveform, cfg, lang, duration_s)
                return text, segments, None, "qwen3-asr"
            if backend == "sensevoice":
                text, segments = _transcribe_sensevoice(waveform, cfg, lang, duration_s)
                return text, segments, None, "sensevoice"
            text, segments, mean_conf = _transcribe_whisper(
                waveform, cfg, lang, warnings, segment_cb
            )
            return text, segments, mean_conf, "faster-whisper"
        except Exception as exc:
            warnings.append(f"{backend} ASR unavailable/failed ({exc}).")
    warnings.append(
        "All ASR backends unavailable; transcript empty. Acoustic biomarkers "
        "were still extracted. Install an ASR extra or pass --transcript."
    )
    return "", [], None, "none"


def _backend_chain(asr_backend: str, lang: str) -> list[str]:
    if asr_backend and asr_backend != "auto":
        return [asr_backend]
    if lang in _CHAIN_LANGUAGES:
        return ["qwen3", "sensevoice", "whisper"]
    return ["whisper"]


def _prepare_waveform(speech: np.ndarray, sr: int) -> np.ndarray:
    waveform = np.asarray(speech, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.float32, copy=False)
    if sr != TARGET_SR:
        waveform = resample_to(waveform, sr, TARGET_SR)
    return np.ascontiguousarray(waveform, dtype=np.float32)


# --- Qwen3-ASR (Apache-2.0; best spoken-Cantonese fidelity) -------------------

def _transcribe_qwen3(
    waveform: np.ndarray, cfg: WhisperConfig, lang: str, duration_s: float
) -> tuple[str, list[TranscriptSegment]]:
    model = _load_qwen3(cfg)
    qwen_lang = _QWEN_LANGUAGE_NAMES.get(lang)
    results = model.transcribe(audio=(waveform, TARGET_SR), language=qwen_lang)
    text = str(results[0].text).strip() if results else ""
    segments = (
        [TranscriptSegment(start_s=0.0, end_s=round(duration_s, 3), text=text)]
        if text
        else []
    )
    return text, segments


def _load_qwen3(cfg: WhisperConfig):
    key = f"qwen3|{cfg.qwen_model}|{cfg.device}"
    if key not in _model_cache:
        import torch  # optional dep: pip install "mci-biomarker[asr-qwen]"
        from qwen_asr import Qwen3ASRModel

        on_cpu = cfg.device.startswith("cpu")
        _model_cache[key] = Qwen3ASRModel.from_pretrained(
            cfg.qwen_model,
            dtype=torch.float32 if on_cpu else torch.bfloat16,
            device_map=cfg.device,
            max_new_tokens=1024,
        )
    return _model_cache[key]


# --- SenseVoice / FunASR (fast; native Cantonese) -----------------------------

def _transcribe_sensevoice(
    waveform: np.ndarray, cfg: WhisperConfig, lang: str, duration_s: float
) -> tuple[str, list[TranscriptSegment]]:
    model = _load_sensevoice(cfg)
    sv_lang = "yue" if lang in _CHAIN_LANGUAGES else ("zh" if is_cjk(lang) else "auto")
    # FunASR most reliably accepts a file path; we write our already-decoded
    # waveform (deleted right after — zero-retention).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        write_wav(tmp.name, waveform, TARGET_SR)
        res = model.generate(
            input=tmp.name, cache={}, language=sv_lang, use_itn=True, batch_size_s=60
        )
    if not res:
        return "", []
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    sentences = res[0].get("sentence_info") if isinstance(res[0], dict) else None
    if sentences:
        segments = [
            TranscriptSegment(
                start_s=float(s.get("start", 0)) / 1000.0,
                end_s=float(s.get("end", 0)) / 1000.0,
                text=rich_transcription_postprocess(str(s.get("text", ""))).strip(),
            )
            for s in sentences
        ]
        return " ".join(s.text for s in segments).strip(), segments
    text = rich_transcription_postprocess(str(res[0].get("text", ""))).strip()
    segments = (
        [TranscriptSegment(start_s=0.0, end_s=round(duration_s, 3), text=text)]
        if text
        else []
    )
    return text, segments


def _load_sensevoice(cfg: WhisperConfig):
    key = f"sensevoice|{cfg.sensevoice_model}|{cfg.device}"
    if key not in _model_cache:
        from funasr import AutoModel  # optional dep: ...[asr-sensevoice]

        _model_cache[key] = AutoModel(
            model=cfg.sensevoice_model,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=cfg.device,
            disable_update=True,
        )
    return _model_cache[key]


# --- faster-whisper ------------------------------------------------------------

def _transcribe_whisper(
    waveform: np.ndarray,
    cfg: WhisperConfig,
    lang: str,
    warnings: list[str],
    segment_cb=None,
) -> tuple[str, list[TranscriptSegment], float | None]:
    model_name = cfg.model
    whisper_lang: str | None = lang if lang != "en" else "en"
    if lang in _CHAIN_LANGUAGES:
        whisper_lang = "yue"  # yue token exists only in large-v3-generation models
        if model_name.endswith(".en"):
            warnings.append(
                f"Whisper model {model_name!r} is English-only; switched to "
                f"{cfg.yue_whisper_model!r} for Cantonese."
            )
            model_name = cfg.yue_whisper_model
    elif lang == "zh":
        whisper_lang = "zh"
    model = _load_whisper_model(model_name, cfg)
    # Pass the decoded waveform, never a file path: faster-whisper's PyAV
    # file decoder calls av.open(..., metadata_errors=...) which newer
    # PyAV versions reject.
    segments_iter, _info = model.transcribe(
        waveform, language=whisper_lang, beam_size=5, vad_filter=False
    )
    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        conf = None
        logprob = getattr(seg, "avg_logprob", None)
        if logprob is not None and math.isfinite(logprob):
            conf = round(float(math.exp(max(-10.0, logprob))), 4)
        segments.append(
            TranscriptSegment(
                start_s=float(seg.start),
                end_s=float(seg.end),
                text=str(seg.text).strip(),
                confidence=conf,
            )
        )
        if segment_cb is not None:
            segment_cb(len(segments))
    transcript = " ".join(s.text for s in segments).strip()
    confs = [s.confidence for s in segments if s.confidence is not None]
    mean_conf = round(float(np.mean(confs)), 4) if confs else None
    return transcript, segments, mean_conf


def _load_whisper_model(model_name: str, cfg: WhisperConfig):
    key = f"whisper|{model_name}|{cfg.device}|{cfg.compute_type}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel  # optional dep

        _model_cache[key] = WhisperModel(
            model_name, device=cfg.device, compute_type=cfg.compute_type
        )
    return _model_cache[key]
