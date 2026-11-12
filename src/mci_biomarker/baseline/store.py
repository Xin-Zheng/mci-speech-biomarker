"""Layer 3: longitudinal store + personal-baseline anomaly detection (SPEC §2).

Profiles persist as one JSONL file per user under `data/profiles/<user>.jsonl`
(JSON only — never audio, SPEC §4). A new session is compared to the user's
prior sessions with rolling z-scores; metrics beyond ±2σ are flagged. With
fewer than `min_baseline_sessions` priors the baseline is reported as
"forming" and no flags are raised (SPEC: 30–90 day baseline window).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..schemas import BaselineComparison, BiomarkerProfile, Patient

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


class VoiceprintStore:
    """Per-patient voiceprint centroids (for speaker auto-matching).

    Stored next to patients.json as mean MFCC/pitch embeddings — derived
    features, not audio (speech cannot be reconstructed from them), but
    still biometric identifiers: the file is git-ignored like profiles.
    """

    MAX_ENROLLMENTS = 20  # rolling centroid over the most recent sessions

    def __init__(self, profile_dir: str | Path):
        self.path = Path(profile_dir).parent / "voiceprints.json"

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    def enroll(self, patient_id: str, embedding) -> None:
        import numpy as np

        patient_id = safe_user_id(patient_id)
        vec = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return
        vec = vec / norm
        data = self._load()
        rec = data.get(patient_id)
        if rec and len(rec.get("centroid", [])) == len(vec):
            count = min(int(rec.get("count", 1)), self.MAX_ENROLLMENTS)
            centroid = (np.asarray(rec["centroid"]) * count + vec) / (count + 1)
            count += 1
        else:
            centroid, count = vec, 1
        cnorm = float(np.linalg.norm(centroid))
        if cnorm > 1e-9:
            centroid = centroid / cnorm
        data[patient_id] = {
            "centroid": [round(float(x), 6) for x in centroid.tolist()],
            "count": count,
        }
        self._save(data)

    def centroids(self) -> dict:
        import numpy as np

        return {
            pid: np.asarray(rec["centroid"], dtype=np.float64)
            for pid, rec in self._load().items()
            if rec.get("centroid")
        }

    def delete(self, patient_id: str) -> bool:
        data = self._load()
        if data.pop(safe_user_id(patient_id), None) is None:
            return False
        self._save(data)
        return True


class PatientStore:
    """Patient registry (display names/notes) next to the profile JSONLs.

    Patients also exist implicitly: any user_id with saved profiles is a
    patient even without a registry entry. `list_patients` merges both.
    """

    def __init__(self, profile_dir: str | Path):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.profile_dir.parent / "patients.json"
        self.profiles = ProfileStore(profile_dir)

    def _load_registry(self) -> dict[str, Patient]:
        if not self.registry_path.exists():
            return {}
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {pid: Patient.model_validate(rec) for pid, rec in raw.items()}

    def _save_registry(self, registry: dict[str, Patient]) -> None:
        payload = {pid: p.model_dump(mode="json") for pid, p in registry.items()}
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def upsert_patient(
        self, patient_id: str, display_name: str = "", notes: str = ""
    ) -> Patient:
        patient_id = safe_user_id(patient_id)
        registry = self._load_registry()
        existing = registry.get(patient_id)
        patient = Patient(
            patient_id=patient_id,
            display_name=display_name or (existing.display_name if existing else patient_id),
            notes=notes if notes else (existing.notes if existing else ""),
            created_at=existing.created_at if existing else Patient(
                patient_id=patient_id
            ).created_at,
        )
        registry[patient_id] = patient
        self._save_registry(registry)
        return patient

    def get_patient(self, patient_id: str) -> Patient | None:
        patient_id = safe_user_id(patient_id)
        registry = self._load_registry()
        if patient_id in registry:
            return registry[patient_id]
        if patient_id in self.profiles.list_users():
            return Patient(patient_id=patient_id, display_name=patient_id)
        return None

    def delete_patient(self, patient_id: str, delete_profiles: bool = False) -> bool:
        patient_id = safe_user_id(patient_id)
        registry = self._load_registry()
        removed = registry.pop(patient_id, None) is not None
        if removed:
            self._save_registry(registry)
        if delete_profiles:
            removed = self.profiles.delete_user(patient_id) or removed
        return removed

    def list_patients(self) -> list[dict]:
        """Registry ∪ profile owners, each with session stats for the UI."""
        registry = self._load_registry()
        ids = sorted(set(registry) | set(self.profiles.list_users()))
        out: list[dict] = []
        for pid in ids:
            patient = registry.get(pid) or Patient(patient_id=pid, display_name=pid)
            profiles = self.profiles.load_profiles(pid)
            latest = profiles[-1] if profiles else None
            out.append(
                {
                    **patient.model_dump(mode="json"),
                    "session_count": len(profiles),
                    "last_session_at": latest.created_at if latest else None,
                    "last_language": latest.linguistic.language if latest else None,
                    "baseline_ready": bool(latest and latest.baseline.baseline_ready),
                    "flagged_count": len(latest.baseline.flagged) if latest else 0,
                }
            )
        return out
