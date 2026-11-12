"""Layer 3: longitudinal store + personal-baseline anomaly detection (SPEC §2).

Profiles persist as one JSONL file per user under `data/profiles/<user>.jsonl`
(JSON only — never audio, SPEC §4). A new session is compared to the user's
prior sessions with rolling z-scores; metrics beyond ±2σ are flagged. With
fewer than `min_baseline_sessions` priors the baseline is reported as
"forming" and no flags are raised (SPEC: 30–90 day baseline window).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..schemas import BaselineComparison, BiomarkerProfile

# Direction notes for the report (higher ≠ worse for every metric).
METRIC_DIRECTIONS = {
    "speech_rate_syll_per_s": "lower = slower speech",
    "articulation_rate_syll_per_s": "lower = slower articulation",
    "mean_pause_duration_s": "higher = longer pauses",
    "pause_rate_per_min": "higher = more frequent pauses",
    "ttr": "lower = reduced lexical diversity",
    "mattr": "lower = reduced lexical diversity",
    "mtld": "lower = reduced lexical diversity",
    "pronoun_to_noun_ratio": "higher = more pronoun substitution",
    "filler_ratio": "higher = more filled pauses",
    "words_per_minute": "lower = slower word output",
}


def safe_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id).strip("._")
    return cleaned or "anonymous"


class ProfileStore:
    def __init__(self, profile_dir: str | Path):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        return self.profile_dir / f"{safe_user_id(user_id)}.jsonl"

    def load_profiles(self, user_id: str) -> list[BiomarkerProfile]:
        path = self._path(user_id)
        if not path.exists():
            return []
        profiles = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                profiles.append(BiomarkerProfile.model_validate_json(line))
        return sorted(profiles, key=lambda p: p.created_at)

    def save_profile(self, profile: BiomarkerProfile) -> Path:
        path = self._path(profile.user_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(profile.model_dump_json() + "\n")
        return path

    def list_users(self) -> list[str]:
        return sorted(p.stem for p in self.profile_dir.glob("*.jsonl"))

    def delete_user(self, user_id: str) -> bool:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
            return True
        return False


def compare_to_baseline(
    profile: BiomarkerProfile,
    priors: list[BiomarkerProfile],
    min_sessions: int = 5,
    z_threshold: float = 2.0,
) -> BaselineComparison:
    """Rolling z-scores of headline metrics vs prior sessions for this user."""
    # Never mix languages in one baseline: lexical metrics are not comparable
    # across e.g. English and Cantonese sessions.
    priors = [p for p in priors if p.linguistic.language == profile.linguistic.language]
    current = profile.headline_metrics()
    if len(priors) < min_sessions:
        return BaselineComparison(
            n_prior_sessions=len(priors),
            baseline_ready=False,
            summary=(
                f"Baseline forming: {len(priors)}/{min_sessions} sessions collected. "
                "Keep recording regularly (SPEC: 30–90 days) before trend flags activate."
            ),
        )
    history: dict[str, list[float]] = {}
    for prior in priors:
        for key, val in prior.headline_metrics().items():
            if val is not None:
                history.setdefault(key, []).append(float(val))
    z_scores: dict[str, float] = {}
    flagged: list[str] = []
    for key, val in current.items():
        vals = history.get(key, [])
        if val is None or len(vals) < 3:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        std = math.sqrt(var)
        if std <= 1e-9:
            continue
        z = (float(val) - mean) / std
        z_scores[key] = round(z, 3)
        if abs(z) >= z_threshold:
            flagged.append(key)
    if flagged:
        details = "; ".join(
            f"{k} z={z_scores[k]:+.2f} ({METRIC_DIRECTIONS.get(k, '')})" for k in flagged
        )
        summary = (
            f"{len(flagged)} biomarker(s) deviate beyond ±{z_threshold:.0f}σ from personal "
            f"baseline: {details}. Treat as a wellness-trend prompt for clinical follow-up, "
            "not a diagnosis."
        )
    else:
        summary = "All headline biomarkers within personal-baseline range this session."
    return BaselineComparison(
        n_prior_sessions=len(priors),
        baseline_ready=True,
        z_scores=z_scores,
        flagged=flagged,
        summary=summary,
    )


def trend_series(profiles: list[BiomarkerProfile]) -> dict:
    """Chart-ready time series for the web UI."""
    labels = [p.created_at[:19].replace("T", " ") for p in profiles]
    metrics: dict[str, list] = {}
    for p in profiles:
        for key, val in p.headline_metrics().items():
            metrics.setdefault(key, []).append(val)
    return {"labels": labels, "metrics": metrics, "directions": METRIC_DIRECTIONS}
