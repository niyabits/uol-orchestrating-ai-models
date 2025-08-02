"""Lightweight affect analysis tools tailored for narrative authoring workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    evidence: List[str]


@dataclass(frozen=True)
class EmotionResult:
    dominant: str
    scores: Dict[str, float]
    evidence: Dict[str, List[str]]


class TextSentimentClassifier:
    """Rule-based sentiment analysis tuned for creative writing snippets."""

    _POSITIVE: Dict[str, float] = {
        "uplift": 1.6,
        "hope": 1.5,
        "hopeful": 1.7,
        "serene": 1.4,
        "serenity": 1.6,
        "calm": 1.2,
        "gentle": 1.3,
        "grace": 1.5,
        "resilient": 1.3,
        "victory": 1.8,
        "triumph": 1.9,
        "warm": 1.3,
        "glow": 1.2,
        "support": 1.1,
        "wonder": 1.2,
        "comfort": 1.4,
        "relief": 1.5,
        "peace": 1.5,
        "freedom": 1.5,
        "curiosity": 1.1,
        "inspired": 1.6,
        "inspiration": 1.7,
        "delight": 1.8,
        "joy": 2.0,
        "promise": 1.6,
        "sacred": 1.2,
        "guide": 1.0,
        "protect": 1.1,
        "steady": 1.0,
        "mastery": 1.6,
    }

    _NEGATIVE: Dict[str, float] = {
        "fear": 1.8,
        "fearful": 1.8,
        "afraid": 1.7,
        "paranoia": 1.9,
        "paranoid": 1.9,
        "dread": 1.8,
        "distant": 0.9,
        "lonely": 1.4,
        "isolation": 1.6,
        "isolated": 1.6,
        "threat": 1.7,
        "danger": 1.6,
        "risk": 1.1,
        "exposed": 1.3,
        "storm": 1.0,
        "violent": 1.9,
        "envy": 1.6,
        "envious": 1.6,
        "jealous": 1.4,
        "anger": 1.7,
        "resent": 1.7,
        "wreck": 1.5,
        "ruin": 1.6,
        "loss": 1.6,
        "betray": 1.8,
        "bitter": 1.5,
        "anxious": 1.6,
        "anxiety": 1.7,
        "exhausted": 1.4,
        "tension": 1.2,
    }

    _INTENSIFIERS: Dict[str, float] = {
        "very": 1.4,
        "deeply": 1.5,
        "incredibly": 1.6,
        "utterly": 1.6,
        "profoundly": 1.6,
        "so": 1.2,
        "remarkably": 1.4,
    }

    _NEGATORS: Sequence[str] = (
        "not",
        "never",
        "hardly",
        "barely",
        "scarcely",
    )

    def analyze(self, text: str) -> SentimentResult:
        tokens = [tok.lower() for tok in _TOKENIZE(text)]
        contributions: List[Tuple[str, float]] = []
        for idx, token in enumerate(tokens):
            base = 0.0
            if token in self._POSITIVE:
                base = self._POSITIVE[token]
            elif token in self._NEGATIVE:
                base = -self._NEGATIVE[token]
            if base == 0.0:
                continue
            scale = 1.0
            prev = tokens[idx - 1] if idx > 0 else None
            if prev in self._INTENSIFIERS:
                scale *= self._INTENSIFIERS[prev]
            if prev in self._NEGATORS:
                scale *= -1.0
            if idx > 1 and tokens[idx - 2] in self._NEGATORS:
                scale *= -1.0
            contributions.append((token, base * scale))

        if not contributions:
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                evidence=[],
            )

        total = sum(weight for _, weight in contributions)
        pos_total = sum(max(weight, 0.0) for _, weight in contributions)
        neg_total = sum(-min(weight, 0.0) for _, weight in contributions)
        magnitude = pos_total + neg_total
        positive_score = pos_total / magnitude if magnitude else 0.0
        negative_score = neg_total / magnitude if magnitude else 0.0
        neutral_score = 1.0 - (positive_score + negative_score)
        neutral_score = max(0.0, neutral_score)

        if magnitude < 1.2 or math.isclose(total, 0.0, abs_tol=0.2):
            label = "neutral"
        elif total > 0:
            label = "positive"
        else:
            label = "negative"

        confidence = min(1.0, magnitude / 4.0)
        ranked_terms = [term for term, _ in sorted(contributions, key=lambda kv: -abs(kv[1]))][:6]

        scores = {
            "positive": round(positive_score, 4),
            "negative": round(negative_score, 4),
            "neutral": round(neutral_score, 4),
        }
        return SentimentResult(label=label, confidence=round(confidence, 4), scores=scores, evidence=ranked_terms)


class EmotionTagger:
    """Tag dominant emotions in narrative passages for downstream media prompts."""

    _LEXICON: Dict[str, Tuple[str, float]] = {}

    _EMOTIONS: Dict[str, Tuple[Sequence[str], float]] = {
        "joy": (
            (
                "joy",
                "delight",
                "delighted",
                "elated",
                "hope",
                "hopeful",
                "warm",
                "glow",
                "laughter",
                "laugh",
                "smile",
                "wonder",
                "awe",
                "serene",
                "serenity",
                "relief",
                "comfort",
                "carefree",
            ),
            1.3,
        ),
        "sadness": (
            (
                "sad",
                "sorrow",
                "sorrowful",
                "melancholy",
                "lonely",
                "loneliness",
                "isolation",
                "isolated",
                "mourning",
                "loss",
                "regret",
                "despair",
                "ache",
                "tear",
                "weary",
            ),
            1.2,
        ),
        "anger": (
            (
                "anger",
                "angry",
                "furious",
                "ire",
                "rage",
                "resent",
                "resentful",
                "bitterness",
                "envy",
                "envious",
                "jealous",
                "seethe",
                "irate",
                "spite",
            ),
            1.3,
        ),
        "fear": (
            (
                "fear",
                "afraid",
                "anxious",
                "anxiety",
                "paranoia",
                "paranoid",
                "terror",
                "dread",
                "threat",
                "danger",
                "tremble",
                "uneasy",
                "wary",
                "worry",
            ),
            1.4,
        ),
    }

    _INTENSIFIERS = TextSentimentClassifier._INTENSIFIERS
    _NEGATORS = TextSentimentClassifier._NEGATORS

    def __init__(self) -> None:
        if not self._LEXICON:
            expanded: Dict[str, Tuple[str, float]] = {}
            for emotion, (terms, base_weight) in self._EMOTIONS.items():
                for term in terms:
                    expanded[term] = (emotion, base_weight)
            type(self)._LEXICON = expanded

    def tag(self, text: str) -> EmotionResult:
        tokens = [tok.lower() for tok in _TOKENIZE(text)]
        scores: Dict[str, float] = {emotion: 0.0 for emotion in self._EMOTIONS}
        evidence: Dict[str, List[str]] = {emotion: [] for emotion in self._EMOTIONS}

        for idx, token in enumerate(tokens):
            if token not in self._LEXICON:
                continue
            emotion, base_weight = self._LEXICON[token]
            scale = base_weight
            prev = tokens[idx - 1] if idx > 0 else None
            if prev in self._INTENSIFIERS:
                scale *= self._INTENSIFIERS[prev]
            negate = False
            if prev in self._NEGATORS or (idx > 1 and tokens[idx - 2] in self._NEGATORS):
                negate = True
            if negate:
                scale *= -0.6
            scores[emotion] += scale
            evidence[emotion].append(token)

        total_activation = sum(max(val, 0.0) for val in scores.values())
        if total_activation == 0.0:
            normalized = {emotion: 0.0 for emotion in scores}
            return EmotionResult(dominant="neutral", scores=normalized, evidence={})

        normalized = {emotion: round(max(val, 0.0) / total_activation, 4) for emotion, val in scores.items()}
        dominant_emotion = max(normalized.items(), key=lambda kv: kv[1])
        dominant = dominant_emotion[0] if dominant_emotion[1] >= 0.34 else "mixed"
        pruned_evidence = {emotion: terms for emotion, terms in evidence.items() if terms}
        return EmotionResult(dominant=dominant, scores=normalized, evidence=pruned_evidence)


def _TOKENIZE(text: str) -> Iterable[str]:
    for match in _TOKEN_RE.finditer(text):
        yield match.group()
