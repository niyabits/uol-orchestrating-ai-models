"""Sentiment-aware voice synthesis planning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .affect import SentimentResult, TextSentimentClassifier


@dataclass(frozen=True)
class VoiceSegment:
    """Voice rendering instructions for a contiguous chunk of narrative."""

    text: str
    sentiment: str
    confidence: float
    rate: float
    pitch_shift: str
    timbre: str
    pace_descriptor: str
    evidence: Sequence[str]
    ssml: str


@dataclass(frozen=True)
class VoiceSynthesisResult:
    """Aggregate voice rendering plan and SSML payload."""

    segments: Sequence[VoiceSegment]
    overall: SentimentResult
    ssml: str
    narration_script: str


class VoiceSynthesisModel:
    """Generate a voice delivery plan that adapts to the story's sentiment."""

    def __init__(self, *, classifier: TextSentimentClassifier | None = None) -> None:
        self._classifier = classifier or TextSentimentClassifier()

    def synthesize(self, text: str, *, speaker: str = "alloy") -> VoiceSynthesisResult:
        """Produce voice instructions and SSML tuned to the text sentiment."""
        cleaned = text.strip()
        if not cleaned:
            empty_sentiment = self._classifier.analyze("")
            empty_segment = VoiceSegment(
                text="",
                sentiment="neutral",
                confidence=0.0,
                rate=1.0,
                pitch_shift="0st",
                timbre="neutral",
                pace_descriptor="steady",
                evidence=[],
                ssml="",
            )
            return VoiceSynthesisResult(
                segments=[empty_segment],
                overall=empty_sentiment,
                ssml=f"<speak><voice name=\"{speaker}\"></voice></speak>",
                narration_script="(silence)",
            )

        paragraphs = _segment_text(cleaned)
        overall_sentiment = self._classifier.analyze(cleaned)

        segments: List[VoiceSegment] = []
        ssml_chunks: List[str] = []
        script_lines: List[str] = []
        for paragraph in paragraphs:
            sentiment = self._classifier.analyze(paragraph)
            profile = _profile_from_sentiment(sentiment)
            escaped = _escape_ssml(paragraph)
            ssml_chunk = (
                f"<p><prosody rate=\"{profile['rate']:.2f}\" pitch=\"{profile['pitch_shift']}\" "
                f"volume=\"{profile['volume']}\">{escaped}</prosody></p>"
            )
            ssml_chunks.append(ssml_chunk)
            script_lines.append(
                f"[{profile['pace_descriptor']}, {profile['timbre']} timbre] {paragraph}"
            )
            segments.append(
                VoiceSegment(
                    text=paragraph,
                    sentiment=sentiment.label,
                    confidence=sentiment.confidence,
                    rate=profile["rate"],
                    pitch_shift=profile["pitch_shift"],
                    timbre=profile["timbre"],
                    pace_descriptor=profile["pace_descriptor"],
                    evidence=sentiment.evidence,
                    ssml=ssml_chunk,
                )
            )

        ssml_payload = f"<speak><voice name=\"{speaker}\">{''.join(ssml_chunks)}</voice></speak>"
        narration_script = "\n".join(script_lines)
        return VoiceSynthesisResult(
            segments=segments,
            overall=overall_sentiment,
            ssml=ssml_payload,
            narration_script=narration_script,
        )


def _segment_text(text: str) -> List[str]:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if blocks:
        return blocks
    return [line.strip() for line in text.splitlines() if line.strip()]


def _profile_from_sentiment(result: SentimentResult) -> dict[str, object]:
    label = result.label or "neutral"
    confidence = max(0.0, min(1.0, result.confidence))

    if label == "positive":
        base_rate = 1.05
        base_pitch = 2.0
        timbre = "bright"
        volume = "loud"
        pace_descriptor = "animated"
    elif label == "negative":
        base_rate = 0.92
        base_pitch = -3.0
        timbre = "warm"
        volume = "medium"
        pace_descriptor = "measured"
    else:
        base_rate = 1.0
        base_pitch = 0.0
        timbre = "neutral"
        volume = "medium"
        pace_descriptor = "steady"

    rate = base_rate + (confidence - 0.5) * 0.12
    rate = max(0.75, min(1.25, rate))

    pitch_shift = base_pitch + (confidence - 0.5) * 3.0
    pitch_shift = max(-7.0, min(7.0, pitch_shift))
    pitch_shift_str = f"{pitch_shift:+.0f}st"

    return {
        "rate": round(rate, 3),
        "pitch_shift": pitch_shift_str,
        "timbre": timbre,
        "volume": volume,
        "pace_descriptor": pace_descriptor,
    }


def _escape_ssml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


__all__ = [
    "VoiceSegment",
    "VoiceSynthesisResult",
    "VoiceSynthesisModel",
]
