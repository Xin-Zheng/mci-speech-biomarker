"""Track B linguistic features (SPEC §2) — pure Python, no model downloads.

Metrics: TTR, MATTR (window 50), MTLD, pronoun-to-noun ratio, vague-word ratio,
filler ratio, immediate-repetition rate, sentence stats, words/minute.
Nouns are estimated by lexicon exclusion (content words that are not pronouns,
vague words, fillers or closed-class function words); if spaCy is installed its
POS tagger refines the noun/pronoun counts automatically.
"""

from __future__ import annotations

import re
from collections import Counter

from ..schemas import LinguisticFeatures, TranscriptSegment

PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those", "who", "whom", "whose", "which",
    "what", "one", "ones", "someone", "something", "anyone", "anything",
    "everyone", "everything", "nobody", "nothing", "somebody", "anybody",
}
VAGUE_WORDS = {
    "thing", "things", "stuff", "whatever", "that", "this", "it",
    "thingy", "whatchamacallit", "whatsit", "somehow", "somewhere",
}
FILLERS = {"um", "uh", "er", "erm", "hmm", "mm", "mhm", "uhh", "umm", "ah", "oh"}
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "when", "while", "because",
    "so", "then", "than", "as", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "about", "into", "over", "after", "before", "under",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "not", "no", "yes",
    "very", "just", "also", "there", "here", "now", "well", "like", "you",
    "know", "mean", "actually", "basically", "really", "right",
}
TOKEN_RE = re.compile(r"[a-z']+")
SENT_RE = re.compile(r"[.!?]+|\n+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def extract_linguistic(
    transcript: str,
    segments: list[TranscriptSegment] | None = None,
    mean_confidence: float | None = None,
    backend: str = "manual",
    speech_duration_s: float | None = None,
    low_confidence_threshold: float = 0.55,
) -> LinguisticFeatures:
    segments = segments or []
    tokens = tokenize(transcript)
    n = len(tokens)
    low_conf = bool(mean_confidence is not None and mean_confidence < low_confidence_threshold)
    noun_count, pronoun_count = _count_nouns_pronouns(transcript, tokens)
    vague = sum(1 for t in tokens if t in VAGUE_WORDS)
    fillers = sum(1 for t in tokens if t in FILLERS)
    sentences = [s for s in SENT_RE.split(transcript) if s.strip()]
    wpm = None
    if speech_duration_s and speech_duration_s > 0:
        wpm = round(n / (speech_duration_s / 60.0), 2)
    return LinguisticFeatures(
        backend=backend,
        transcript=transcript,
        segments=segments,
        mean_confidence=mean_confidence,
        low_confidence=low_conf,
        token_count=n,
        type_count=len(set(tokens)),
        ttr=_r(ttr(tokens)),
        mattr=_r(mattr(tokens)),
        mtld=_r(mtld(tokens)),
        pronoun_count=pronoun_count,
        noun_count=noun_count,
        pronoun_to_noun_ratio=_r(pronoun_count / noun_count) if noun_count else None,
        vague_word_count=vague,
        vague_word_ratio=_r(vague / n) if n else None,
        filler_count=fillers,
        filler_ratio=_r(fillers / n) if n else None,
        repeated_word_rate=_r(_repeated_rate(tokens)),
        sentence_count=len(sentences),
        mean_sentence_length=_r(n / len(sentences)) if sentences else None,
        words_per_minute=wpm,
    )


# --- Lexical diversity -------------------------------------------------------

def ttr(tokens: list[str]) -> float | None:
    return len(set(tokens)) / len(tokens) if tokens else None


def mattr(tokens: list[str], window: int = 50) -> float | None:
    """Moving-average TTR; falls back to TTR for short transcripts."""
    if not tokens:
        return None
    if len(tokens) <= window:
        return ttr(tokens)
    ratios = [
        len(set(tokens[i : i + window])) / window for i in range(len(tokens) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def mtld(tokens: list[str], threshold: float = 0.72) -> float | None:
    """Measure of Textual Lexical Diversity (mean of fwd/bwd passes)."""
    if len(tokens) < 10:
        return None
    fwd = _mtld_pass(tokens, threshold)
    bwd = _mtld_pass(list(reversed(tokens)), threshold)
    vals = [v for v in (fwd, bwd) if v]
    return sum(vals) / len(vals) if vals else None


def _mtld_pass(tokens: list[str], threshold: float) -> float | None:
    factors, seen, count = 0.0, set(), 0
    for tok in tokens:
        seen.add(tok)
        count += 1
        if count and len(seen) / count <= threshold:
            factors += 1
            seen, count = set(), 0
    if count:  # partial factor
        partial_ttr = len(seen) / count
        factors += (1 - partial_ttr) / (1 - threshold) if threshold < 1 else 0
    return len(tokens) / factors if factors > 0 else None


# --- POS-ish counting ---------------------------------------------------------

def _count_nouns_pronouns(transcript: str, tokens: list[str]) -> tuple[int, int]:
    spacy_counts = _spacy_counts(transcript)
    if spacy_counts is not None:
        return spacy_counts
    pronouns = sum(1 for t in tokens if t in PRONOUNS)
    nouns = sum(
        1
        for t in tokens
        if t not in PRONOUNS and t not in FUNCTION_WORDS and t not in FILLERS
    )
    return nouns, pronouns


def _spacy_counts(transcript: str) -> tuple[int, int] | None:
    try:
        import spacy  # optional dep: pip install "mci-biomarker[nlp]"

        if not spacy.util.is_package("en_core_web_sm"):
            return None
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        doc = nlp(transcript)
        nouns = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
        pronouns = sum(1 for t in doc if t.pos_ == "PRON")
        return nouns, pronouns
    except Exception:
        return None


def _repeated_rate(tokens: list[str]) -> float | None:
    if len(tokens) < 2:
        return None
    reps = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)
    return reps / len(tokens)


def top_repetitions(transcript: str, k: int = 5) -> list[tuple[str, int]]:
    """Most repeated content words (perseveration / topic-drift aid)."""
    counts = Counter(
        t for t in tokenize(transcript) if t not in FUNCTION_WORDS and len(t) > 2
    )
    return counts.most_common(k)


def _r(x: float | None, nd: int = 4) -> float | None:
    return round(float(x), nd) if x is not None else None
