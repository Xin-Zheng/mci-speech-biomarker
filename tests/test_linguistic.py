"""Linguistic feature tests with known expected values."""

from mci_biomarker.linguistic.features import (
    extract_linguistic,
    mattr,
    mtld,
    tokenize,
    ttr,
)

RICH = (
    "The mother washes dishes while water overflows from the sink. "
    "A boy on a stool reaches into the cookie jar as his sister watches. "
    "Plates and cups sit on the counter beside the open window."
)
VAGUE = "um it is that thing and that thing um it that thing there um it"


def test_tokenize_lowercases_and_strips_punct():
    assert tokenize("Hello, WORLD! It's me.") == ["hello", "world", "it's", "me"]


def test_ttr_and_mattr():
    toks = tokenize(RICH)
    assert ttr(toks) == len(set(toks)) / len(toks)
    m = mattr(toks, window=10)
    assert m is not None and 0 < m <= 1


def test_mtld_rich_beats_repetitive():
    rich_score = mtld(tokenize(" ".join([RICH] * 2)))
    rep_score = mtld(tokenize("cat " * 60))
    assert rich_score is not None and rep_score is not None
    assert rich_score > rep_score


def test_pronoun_ratio_and_fillers_detect_vague_speech():
    rich = extract_linguistic(RICH, backend="manual")
    vague = extract_linguistic(VAGUE, backend="manual")
    assert vague.pronoun_to_noun_ratio is not None
    assert rich.pronoun_to_noun_ratio is not None
    assert vague.pronoun_to_noun_ratio > rich.pronoun_to_noun_ratio
    assert vague.filler_count >= 3
    assert vague.vague_word_count >= 5


def test_low_confidence_flag():
    feats = extract_linguistic(RICH, mean_confidence=0.3, backend="faster-whisper")
    assert feats.low_confidence is True
    feats = extract_linguistic(RICH, mean_confidence=0.9, backend="faster-whisper")
    assert feats.low_confidence is False
