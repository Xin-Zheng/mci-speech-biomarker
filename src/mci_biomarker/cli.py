"""`mci-analyze` CLI: wav → structured JSON biomarker profile."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .config import PipelineConfig
from .pipeline import analyze_wav


@click.command()
@click.argument("wav", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--user-id", default="anonymous", show_default=True, help="Longitudinal user id.")
@click.option("--transcript", default=None, help="Known transcript (skips ASR).")
@click.option("--language", default="en", show_default=True,
              help="Session language: en, or yue for Cantonese mode (Qwen3-ASR → "
              "SenseVoice → Whisper-yue + Cantonese linguistic metrics).")
@click.option(
    "--asr-backend",
    type=click.Choice(["auto", "qwen3", "sensevoice", "whisper"]),
    default="auto",
    show_default=True,
    help="Force an ASR engine instead of the language-aware chain.",
)
@click.option("--whisper-model", default=None, help="Override Whisper model, e.g. small, medium.")
@click.option(
    "--diarization",
    type=click.Choice(["auto", "pyannote", "energy"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--acoustic-backend",
    type=click.Choice(["auto", "opensmile", "signal"]),
    default="auto",
    show_default=True,
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Write JSON profile to this file (also prints unless --quiet).")
@click.option("--save/--no-save", default=False, show_default=True,
              help="Append profile to the longitudinal store (data/profiles/).")
@click.option("--pretty/--compact", default=False, help="Pretty-print JSON.")
@click.option("--quiet", is_flag=True, help="Suppress stdout JSON (use with --out).")
def main(
    wav: Path,
    user_id: str,
    transcript: str | None,
    language: str,
    asr_backend: str,
    whisper_model: str | None,
    diarization: str,
    acoustic_backend: str,
    out: Path | None,
    save: bool,
    pretty: bool,
    quiet: bool,
) -> None:
    """Analyze WAV and emit a structured JSON biomarker profile."""
    cfg = PipelineConfig()
    cfg.language = language
    cfg.whisper.asr_backend = asr_backend
    cfg.diarization = diarization
    # extract_acoustic: "auto"/"opensmile" try openSMILE; anything else → signal fallback
    cfg.acoustic_backend = acoustic_backend
    if whisper_model:
        cfg.whisper.model = whisper_model
    profile = analyze_wav(
        wav,
        user_id=user_id,
        cfg=cfg,
        transcript_override=transcript,
        language=language,
        save=save,
    )
    payload = json.dumps(profile.model_dump(mode="json"), indent=2 if pretty else None)
    if out:
        out.write_text(payload + "\n", encoding="utf-8")
        click.echo(f"Profile written to {out}", err=True)
    if not quiet:
        click.echo(payload)
    if profile.provenance.warnings:
        for w in profile.provenance.warnings:
            click.echo(f"warning: {w}", err=True)


if __name__ == "__main__":
    main()
