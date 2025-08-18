"""Sentiment-aware voice synthesis planning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import io
import math
import wave

import numpy as np

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


@dataclass(frozen=True)
class VoiceAudioResult:
    """PCM waveform derived from the synthesis plan."""

    sample_rate: int
    waveform: Any  # numpy.ndarray
    duration_seconds: float

    def to_wav_bytes(self) -> bytes:
        """Serialise the waveform to 16-bit PCM WAV bytes."""

        clipped = np.clip(self.waveform, -1.0, 1.0)
        pcm = np.int16(clipped * 32767)
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm.tobytes())
            return buffer.getvalue()


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

    def render_audio(
        self,
        result: VoiceSynthesisResult,
        *,
        sample_rate: int = 22050,
    ) -> VoiceAudioResult:
        """Convert the synthesis plan into a simple prosody-aware waveform."""

        if not result.segments:
            empty = np.zeros((int(sample_rate * 0.5),), dtype=np.float32)
            return VoiceAudioResult(sample_rate=sample_rate, waveform=empty, duration_seconds=0.5)

        segments: List[np.ndarray] = []
        silence = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
        for segment in result.segments:
            tone = _render_segment_audio(segment, sample_rate=sample_rate)
            segments.append(tone)
            segments.append(silence)

        if segments:
            waveform = np.concatenate(segments)
        else:
            waveform = np.zeros((int(sample_rate * 0.5),), dtype=np.float32)

        duration = waveform.shape[0] / sample_rate
        return VoiceAudioResult(sample_rate=sample_rate, waveform=waveform, duration_seconds=duration)


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


def _render_segment_audio(segment: VoiceSegment, *, sample_rate: int) -> np.ndarray:
    """Create a simple tone sketch that reflects segment sentiment."""

    words = max(1, len(segment.text.split()))
    base_duration = max(0.8, words * 0.26)
    confidence = max(0.0, min(1.0, segment.confidence or 0.0))

    if segment.sentiment == "positive":
        base_freq = 210.0
        brightness = 0.55
    elif segment.sentiment == "negative":
        base_freq = 150.0
        brightness = 0.35
    else:
        base_freq = 180.0
        brightness = 0.45

    pitch_shift = _parse_pitch_shift(segment.pitch_shift)
    freq = base_freq * (2.0 ** (pitch_shift / 12.0))
    sustain = max(0.35, min(0.9, confidence + 0.2))

    t = np.linspace(0.0, base_duration, int(sample_rate * base_duration), endpoint=False)
    modulation = 1.0 + 0.18 * np.sin(2 * math.pi * 1.2 * t)
    carrier = np.sin(2 * math.pi * freq * t * modulation)

    harmonic = brightness * np.sin(2 * math.pi * freq * 2 * t)
    waveform = carrier * (1.0 - brightness) + harmonic

    attack = int(sample_rate * 0.12)
    release = int(sample_rate * 0.18)
    sustain_length = waveform.size - attack - release
    if sustain_length < 0:
        sustain_length = 0
    envelope = np.concatenate(
        [
            np.linspace(0.0, 1.0, max(1, attack), endpoint=False),
            np.ones(max(1, sustain_length)) * sustain,
            np.linspace(sustain, 0.0, max(1, release), endpoint=False),
        ]
    )
    envelope = envelope[: waveform.size]
    amplitude = 0.5 + 0.4 * confidence
    audio = waveform * envelope * amplitude
    return audio.astype(np.float32)


def _parse_pitch_shift(value: str) -> float:
    if not value:
        return 0.0
    cleaned = value.strip().lower().replace("st", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


__all__ = [
    "VoiceSegment",
    "VoiceSynthesisResult",
    "VoiceSynthesisModel",
    "VoiceAudioResult",
]
