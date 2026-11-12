Project Specification: Digital Speech Biomarker Monitoring System for MCI (Mild Cognitive Impairment)
1. Project Context & Objective
Build a longitudinal, non-intrusive software system to detect early signs of Mild Cognitive Impairment (MCI) through everyday speech analysis. Cognitive decline leaves measurable acoustic and linguistic footprints—such as slower speech rate, increased pauses, reduced lexical diversity, nominalization/pronoun substitution (e.g., "that thing"), topic drift, and repetitive questioning.
The system must capture speech passively or via lightweight interactive tasks, extract dual-track biomarkers locally for privacy, and track longitudinal deviations against an individual's personal baseline.

2. System Architecture (4-Layer Pipeline)
Layer 1: Edge Ingestion & Privacy-First Capture
Collection: create a web UI to let user upload the the audio recording
Layer 2: Dual-Track Feature Extraction Engine
Track A: Acoustic / Paralinguistic Features
Extract Low-Level Descriptors (LLDs) and functionals using openSMILE (eGeMAPS / ComParE feature sets) and Parselmouth (Praat).
Key Metrics: Speech rate (syllables/sec), articulation rate, silent/unvoiced pause frequency and duration, fundamental frequency (F₀) perturbation, jitter, and shimmer.
Track B: Linguistic / Semantic Features
ASR Transcription: Local speech-to-text using Whisper (e.g., faster-whisper) or fine-tuned dialect models.
Lexical Diversity: Type-Token Ratio (TTR), Moving-Average Type-Token Ratio (MATTR), and MTLD.
Referential Vagueness: Ratio of pronouns/demonstratives ("that", "it", "thing") to concrete nouns.
Syntactic Complexity: Dependency parse tree depth and clause density.
Semantic Coherence & Repetition: Use embeddings or a lightweight LLM to measure cosine similarity across conversation turns to flag topic drift or repeated questions within short time windows.
Layer 3: Personal Baseline & Longitudinal Anomaly Detection
Dynamic Baseline: Single-point snapshots fail due to daily fatigue, mood, or dialect variance. Aggregate data over the first 30–90 days to build a personalized healthy vocal baseline for each user.
Time-Series Anomaly Detection: Compute rolling Z-scores or use time-series anomaly detection (e.g., Isolation Forest, Prophet, or EWMA). Trigger an alert only when key biomarkers (e.g., mean pause duration + pronoun ratio) deviate by >2σ from the user's personal baseline persistently over multiple days/weeks.
Layer 4: Clinical Decision Support & Reporting
Non-Diagnostic Framing: Position alerts as "Cognitive Wellness Trend Reports" rather than medical diagnoses.
Actionable Output: Generate structured PDF/JSON reports showing longitudinal biomarker trendlines to assist neurologists during formal MMSE/MoCA assessments or biomarker testing (e.g., Aβ PET, CSF). and create a web UI to show each users’ biomarker trend and report

3. Recommended Open-Source Stack & Libraries
Core Audio & Feature Extraction Libraries
opensmile (opensmile-python): Industry standard for extracting ComParE and eGeMAPS acoustic feature sets.
praat-parselmouth: Python interface to Praat for precise measurement of speech rate, pauses, formants, and pitch (F₀).
pyannote.audio: Speaker diarization and VAD to isolate the target user's voice from TV background noise or family members.
speechbrain: PyTorch-based toolkit for ASR, speaker verification, and paralinguistic embeddings.
Reference MCI / Dementia Detection Repositories
mci_detection (://github.com): Multimodal MCI detection framework combining linguistic disfluency features with eGeMAPS acoustic features (benchmarked on Pitt, ADReSS, and TAUKADIAL datasets).
Linguistic_feature_extraction (://github.com): Extracts interpretable linguistic metrics (lexical richness, POS tags, surface features) for cognitive decline corpora.
PROCESS-2 (://github.com): Open-source benchmark pipeline for dementia/MCI speech biomarker analysis.

4. Engineering Constraints & Edge Cases to Handle
Dialect & Accent Robustness: Standard ASR models degrade heavily on regional dialects/accents, corrupting downstream NLP metrics. Ensure the pipeline weights pure acoustic features heavily if ASR confidence scores drop below a threshold, or support fine-tuned Whisper checkpoints.
Noise & Multi-Speaker Filtering: Home environments contain TV audio, kitchen noise, and overlapping speech. Enforce strict VAD + Speaker Verification (pyannote / SpeechBrain) before running feature extraction.
Zero-Retention Audio Policy: Design the local storage schema so audio buffers exist only in volatile memory (RAM) or encrypted temporary storage that is wiped immediately upon feature vector serialization.
