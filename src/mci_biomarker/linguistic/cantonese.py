"""Cantonese (yue) linguistic pack: segmentation + lexicons.

Written Cantonese has no spaces, so Track B metrics need word segmentation
first. We use PyCantonese when installed (proper Cantonese segmenter built on
HKCanCor), and otherwise fall back to greedy longest-match against the
lexicons below + a common-word list, with single characters as last resort.
English code-switched words (very common in HK Cantonese) are kept as tokens.

Lexicons cover both colloquial Cantonese characters (佢/嘅/哋/嘢) and the
standard written forms (他/的/們/東西) that some ASR engines emit instead.
"""

from __future__ import annotations

import re

CJK_LANGUAGES = ("yue", "zh", "cantonese", "zh-hk", "zh-tw", "zh-cn")

PRONOUNS_YUE = {
    # Colloquial Cantonese
    "我", "你", "佢", "我哋", "你哋", "佢哋", "我地", "你地", "佢地",
    "呢個", "嗰個", "呢啲", "嗰啲", "呢度", "嗰度", "邊個", "邊度",
    # Written-Chinese forms some ASR engines substitute
    "他", "她", "它", "我們", "你們", "他們", "她們", "這", "那",
    "這個", "那個", "這些", "那些", "誰", "什么", "甚麼",
}
VAGUE_YUE = {
    "嘢", "嗰樣嘢", "嗰啲嘢", "乜嘢", "樣嘢", "個嘢", "呢樣嘢", "嗰種嘢",
    "東西", "那個東西", "什麼的", "等等",
}
FILLERS_YUE = {
    "嗯", "誒", "呃", "啊", "唔", "即係", "即系", "嗰個", "呢個",
    "咁樣", "點講呢", "係咁", "嗱", "就是", "那個", "這個",
}
FUNCTION_YUE = {
    # Colloquial particles / function words
    "嘅", "喺", "係", "唔係", "有", "冇", "同", "同埋", "俾", "被", "將",
    "緊", "住", "咗", "過", "先", "至", "就", "都", "又", "亦", "啦", "喇",
    "咩", "呀", "嘛", "呢", "嗰", "嘅啫", "咗啦", "而家", "跟住", "之後",
    "因為", "所以", "但係", "不過", "如果", "已經", "仲", "重", "好",
    # Written forms
    "的", "了", "是", "在", "和", "跟", "也", "不", "沒有", "著", "把",
    "很", "就", "還", "現在", "然後", "但是", "一個", "個",
}
# Common content words (segmentation aid; picture-description friendly).
COMMON_YUE_WORDS = {
    "媽媽", "爸爸", "細路", "小朋友", "男仔", "女仔", "哥哥", "姐姐",
    "弟弟", "妹妹", "老人家", "婆婆", "爺爺", "先生", "太太",
    "洗碗", "洗手", "煮飯", "食飯", "飲水", "瞓覺", "起身", "返工",
    "返學", "行路", "坐低", "企喺", "企起身", "攞住", "攞嘢", "打開",
    "閂埋", "跌咗", "跌落", "倒瀉", "流出嚟", "睇住", "望住", "講嘢",
    "水喉", "星盆", "洗手盆", "曲奇", "餅乾", "餅", "罐", "櫃", "雪櫃",
    "凳", "檯", "枱", "窗", "門口", "廚房", "廁所", "客廳", "睡房",
    "盤", "碟", "碗", "杯", "筷子", "匙羹", "地板", "地下", "上面",
    "下面", "隔籬", "旁邊", "後面", "前面", "入面", "出面", "水",
    "禮物", "書包", "電話", "電視", "電腦", "車", "巴士", "地鐵",
    "醫院", "學校", "公園", "街市", "超市", "今日", "聽日", "琴日",
    "朝早", "晏晝", "夜晚", "朝頭早", "開心", "唔開心", "攰", "病",
    "食藥", "睇醫生", "記得", "唔記得", "知道", "唔知", "諗住", "想要",
}
_VOCAB = PRONOUNS_YUE | VAGUE_YUE | FILLERS_YUE | FUNCTION_YUE | COMMON_YUE_WORDS
_MAX_WORD_LEN = max(len(w) for w in _VOCAB)
_LATIN_RE = re.compile(r"[a-zA-Z']+|\d+")
_PUNCT_RE = re.compile(r"[\s，。！？、；：「」『』（）【】《》…·,.!?;:'\"()\[\]<>~—\-]+")


def is_cjk(language: str | None) -> bool:
    return bool(language) and language.lower().strip() in CJK_LANGUAGES


def tokenize_cjk(text: str) -> list[str]:
    """Segment Cantonese/Chinese text into word tokens (code-switch safe)."""
    segmented = _pycantonese_segment(text)
    if segmented is not None:
        return segmented
    return _maxmatch_segment(text)


def _pycantonese_segment(text: str) -> list[str] | None:
    try:
        import pycantonese  # optional dep: pip install "mci-biomarker[cantonese]"

        words = pycantonese.segment(text)
        tokens = [w.strip().lower() for w in words if w and not _PUNCT_RE.fullmatch(w)]
        return tokens or None
    except Exception:
        return None


def _maxmatch_segment(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _PUNCT_RE.match(ch):
            i += 1
            continue
        latin = _LATIN_RE.match(text, i)
        if latin:
            tokens.append(latin.group(0).lower())
            i = latin.end()
            continue
        matched = None
        for length in range(min(_MAX_WORD_LEN, n - i), 1, -1):
            cand = text[i : i + length]
            if cand in _VOCAB:
                matched = cand
                break
        if matched:
            tokens.append(matched)
            i += len(matched)
        else:
            tokens.append(ch)
            i += 1
    return tokens
