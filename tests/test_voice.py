from __future__ import annotations

from mmstory.voice import VoiceSynthesisModel
from app import generate_story_bundle, DEFAULT_CHARACTERS


def test_voice_model_respects_sentiment_shift() -> None:
    model = VoiceSynthesisModel()

    positive = model.synthesize("Joy and triumph glow with hopeful warmth.")
    assert positive.overall.label == "positive"
    assert positive.segments[0].pitch_shift.startswith("+")

    negative = model.synthesize("Fear and dread betray ruin and loss.")
    assert negative.overall.label == "negative"
    assert negative.segments[0].pitch_shift.startswith("-")
    assert "prosody" in negative.segments[0].ssml


def test_story_bundle_includes_voice_outputs() -> None:
    result = generate_story_bundle(
        seed_prompt="A musician learns that harmonies can bend time.",
        beat_index=1,
        target_length=120,
        characters=DEFAULT_CHARACTERS,
        scene_goal="Introduce the temporal rift through music.",
        conflict_axis="ambition vs restraint",
        turns=4,
        style_directive="",
        enable_images=False,
        image_seed="",
        portrait_focus=DEFAULT_CHARACTERS[0][0],
        portrait_wardrobe="stage attire with temporal motifs",
    )

    voice_summary = result[9]
    voice_segments = result[10]
    voice_ssml = result[11]
    voice_script = result[12]

    assert "Overall sentiment" in voice_summary
    assert voice_segments, "voice plan should produce segment rows"
    assert voice_ssml.startswith("<speak><voice")
    assert voice_script, "narration script should not be empty"
