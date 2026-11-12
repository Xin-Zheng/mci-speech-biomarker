"""`mci-analyze` CLI: wav → structured JSON biomarker profile."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .config import PipelineConfig
from .pipeline import analyze_wav, analyze_wav_speakers


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
    type=click.Choice(["auto", "pyannote", "cluster", "energy"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--speakers",
    type=click.Choice(["target", "all"]),
    default="target",
    show_default=True,
    help="target = dominant speaker only; all = one profile per detected speaker.",
)
@click.option("--speaker-map", default=None,
              help='With --speakers all: "SPEAKER_00=chan-tai,SPEAKER_01=lee-mui".')
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
@click.option("--progress/--no-progress", default=True, show_default=True,
              help="Print stage progress to stderr while analyzing.")
def main(
    wav: Path,
    user_id: str,
    transcript: str | None,
    language: str,
    asr_backend: str,
    whisper_model: str | None,
    diarization: str,
    speakers: str,
    speaker_map: str | None,
    acoustic_backend: str,
    out: Path | None,
    save: bool,
    pretty: bool,
    quiet: bool,
    progress: bool,
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

    def progress_cb(pct: int, stage: str) -> None:
        if progress:
            click.echo(f"[{pct:3d}%] {stage}", err=True)

    if speakers == "all":
        mapping = {}
        for pair in (speaker_map or "").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                mapping[k.strip()] = v.strip()
        diar, profiles = analyze_wav_speakers(
            wav,
            base_user_id=user_id,
            cfg=cfg,
            language=language,
            speaker_map=mapping,
            transcripts={"SPEAKER_00": transcript} if transcript else None,
            save=save,
            progress_cb=progress_cb,
        )
        payload_obj: object = {
            "diarization": diar.model_dump(mode="json"),
            "profiles": [p.model_dump(mode="json") for p in profiles],
        }
        all_warnings = [w for p in profiles for w in p.provenance.warnings]
    else:
        profile = analyze_wav(
            wav,
            user_id=user_id,
            cfg=cfg,
            transcript_override=transcript,
            language=language,
            save=save,
            progress_cb=progress_cb,
        )
        payload_obj = profile.model_dump(mode="json")
        all_warnings = profile.provenance.warnings
    payload = json.dumps(payload_obj, indent=2 if pretty else None)
    if out:
        out.write_text(payload + "\n", encoding="utf-8")
        click.echo(f"Profile written to {out}", err=True)
    if not quiet:
        click.echo(payload)
    for w in dict.fromkeys(all_warnings):
        click.echo(f"warning: {w}", err=True)


if __name__ == "__main__":
    main()
