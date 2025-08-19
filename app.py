"""Interactive UI made with Gradio

Users can start from a seed idea, expand it into a brief, drill into a specific
beat, explore dialogue beats, style variations.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List, Optional, Sequence

import gradio as gr

try:  # Pillow is a transitive dependency of diffusers but guard just in case.
    from PIL import Image
except ImportError:  # pragma: no cover - runtime guard only
    Image = None  # type: ignore[assignment]

from mmstory import (
    Character,
    CharacterPortraitGenerator,
    SceneToImageGenerator,
    VoiceSynthesisModel,
    expand_prompt,
    generate_dialogue,
    render_scene_prose,
    scene_json_from_outline,
    style_transfer,
)
from mmstory.image import ImageGenerationResult

CHAR_HEADERS = ["name", "role", "objective", "emotion"]
DEFAULT_CHARACTERS = [
    ["Elizabeth Hawthorne", "Protagonist", "Master the storm-binding dialect", "Determined"],
    ["Brother Ambrose", "Mentor", "Guide Elizabeth without breaking his vows", "Guarded"],
]
VOICE_SEGMENT_HEADERS = [
    "Segment",
    "Sentiment",
    "Confidence",
    "Pace",
    "Pitch",
    "Timbre",
    "Evidence",
]

SCENE_IMAGE_GENERATOR = SceneToImageGenerator()
CHAR_PORTRAIT_GENERATOR = CharacterPortraitGenerator()
VOICE_SYNTHESIZER = VoiceSynthesisModel()


def _rows_to_dicts(rows: Iterable[Any]) -> List[Dict[str, str]]:
    """Normalize character rows coming back from a Gradio dataframe."""
    result: List[Dict[str, str]] = []
    if rows is None:
        return result

    if hasattr(rows, "to_dict"):
        try:
            iterable = rows.to_dict(orient="records")
        except TypeError:
            iterable = rows
    else:
        iterable = rows

    for row in iterable:
        if row is None:
            continue
        if isinstance(row, dict):
            data = {k: str(v).strip() if v is not None else "" for k, v in row.items()}
        else:
            values = list(row)
            values += [""] * (len(CHAR_HEADERS) - len(values))
            data = {k: str(v).strip() for k, v in zip(CHAR_HEADERS, values)}
        if not data.get("name"):
            continue
        result.append({key: data.get(key, "") for key in CHAR_HEADERS})
    return result


def _image_bytes_to_pil(data: Optional[bytes]):
    if data is None:
        return None
    if Image is None:
        raise RuntimeError("Pillow is required to materialise generated images")
    return Image.open(io.BytesIO(data))


def _describe_result(result: ImageGenerationResult) -> str:
    lines: List[str] = [result.prompt.strip()]
    if result.negative_prompt:
        lines.append(f"Negative prompt: {result.negative_prompt.strip()}")
    if result.seed is not None:
        lines.append(f"Seed: {result.seed}")
    if result.metadata:
        meta_bits: List[str] = []
        for key, value in result.metadata.items():
            if value in (None, ""):
                continue
            meta_bits.append(f"{key}={value}")
        if meta_bits:
            lines.append(f"Metadata: {', '.join(meta_bits)}")
    return "\n\n".join([segment for segment in lines if segment])


def _select_portrait_subject(
    char_dicts: Sequence[Dict[str, str]],
    scene_characters,
    desired_name: str,
) -> Optional[Character]:
    name = desired_name.strip().lower()
    if name:
        for character in scene_characters:
            if character.name.lower() == name:
                return character
        for row in char_dicts:
            if row.get("name", "").strip().lower() == name:
                try:
                    return Character.model_validate(row)
                except Exception:
                    pass
    if scene_characters:
        return scene_characters[0]
    if char_dicts:
        try:
            return Character.model_validate(char_dicts[0])
        except Exception:
            return None
    return None


def _parse_seed(raw: str) -> Optional[int]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def generate_story_bundle(
    seed_prompt: str,
    beat_index: int,
    target_length: int,
    characters: Iterable[Any],
    scene_goal: str,
    conflict_axis: str,
    turns: int,
    style_directive: str,
    enable_images: bool,
    image_seed: str,
    portrait_focus: str,
    portrait_wardrobe: str,
) -> tuple[Any, ...]:
    if not seed_prompt.strip():
        raise gr.Error("Please provide a seed concept to expand.")

    brief = expand_prompt(seed_prompt.strip())

    scene = scene_json_from_outline(brief, beat_index=int(beat_index))
    prose = render_scene_prose(scene, target_len=int(target_length))

    char_dicts = _rows_to_dicts(characters)
    dialogue = ""
    if char_dicts:
        if not scene_goal.strip():
            raise gr.Error("Dialogue generation requires a scene goal.")
        if not conflict_axis.strip():
            raise gr.Error("Dialogue generation requires a conflict axis.")
        dialogue = generate_dialogue(
            char_dicts,
            scene_goal.strip(),
            conflict_axis.strip(),
            turns=max(1, int(turns)),
        )

    styled = style_transfer(prose, style_directive.strip()) if style_directive.strip() else ""
    scene_json = scene.model_dump()

    scene_prompt = ""
    scene_image = None
    portrait_prompt = ""
    portrait_image = None
    voice_summary = ""
    voice_segments_table: List[List[str]] = []
    voice_ssml = ""
    voice_script = ""
    voice_audio_data = None
    warnings: List[str] = []

    seed_value = _parse_seed(image_seed)
    if image_seed.strip() and seed_value is None:
        warnings.append("Image seed must be numeric; using random seed instead.")

    primary_voice_text = styled.strip() or prose.strip()
    voice_source_parts: List[str] = []
    if primary_voice_text:
        voice_source_parts.append(primary_voice_text)
    if dialogue.strip():
        voice_source_parts.append(dialogue.strip())
    voice_source = "\n\n".join(part for part in voice_source_parts if part)

    if voice_source:
        try:
            voice_result = VOICE_SYNTHESIZER.synthesize(voice_source)
            scores = voice_result.overall.scores
            voice_summary = (
                f"**Overall sentiment:** {voice_result.overall.label} "
                f"(confidence {voice_result.overall.confidence:.2f})\n\n"
                f"Scores — positive: {scores.get('positive', 0.0):.2f}, "
                f"negative: {scores.get('negative', 0.0):.2f}, "
                f"neutral: {scores.get('neutral', 0.0):.2f}"
            )
            voice_segments_table = [
                [
                    str(idx + 1),
                    segment.sentiment,
                    f"{segment.confidence:.2f}",
                    segment.pace_descriptor,
                    segment.pitch_shift,
                    segment.timbre,
                    ", ".join(segment.evidence),
                ]
                for idx, segment in enumerate(voice_result.segments)
            ]
            voice_ssml = voice_result.ssml
            voice_script = voice_result.narration_script
            try:
                audio_result = VOICE_SYNTHESIZER.render_audio(voice_result)
                voice_audio_data = (audio_result.sample_rate, audio_result.waveform)
            except Exception as audio_exc:
                warnings.append(f"Voice audio rendering failed: {audio_exc}")
        except Exception as exc:
            warnings.append(f"Voice synthesis failed: {exc}")
            voice_summary = "Voice synthesis unavailable."
    else:
        voice_summary = "No narrative content available for voice synthesis."

    if enable_images:
        try:
            scene_result = SCENE_IMAGE_GENERATOR.generate(scene, seed=seed_value)
            scene_prompt = _describe_result(scene_result)
            try:
                scene_image = _image_bytes_to_pil(scene_result.image_bytes)
            except Exception as exc:
                warnings.append(f"Scene image conversion failed: {exc}")
        except Exception as exc:
            warnings.append(f"Scene image generation failed: {exc}")

        portrait_subject = _select_portrait_subject(char_dicts, scene.characters, portrait_focus)
        if portrait_subject is None:
            warnings.append("No viable character found for portrait generation.")
        else:
            try:
                portrait_result = CHAR_PORTRAIT_GENERATOR.generate(
                    portrait_subject,
                    scene=scene,
                    wardrobe=portrait_wardrobe.strip() or None,
                    seed=seed_value,
                )
                portrait_prompt = _describe_result(portrait_result)
                try:
                    portrait_image = _image_bytes_to_pil(portrait_result.image_bytes)
                except Exception as exc:
                    warnings.append(f"Portrait image conversion failed: {exc}")
            except Exception as exc:
                warnings.append(f"Portrait generation failed: {exc}")
    else:
        warnings.append("Image generation disabled.")

    warning_text = "\n".join(warnings)

    return (
        brief,
        scene_json,
        prose,
        dialogue,
        styled,
        scene_prompt,
        scene_image,
        portrait_prompt,
        portrait_image,
        voice_summary,
        voice_segments_table,
        voice_ssml,
        voice_script,
        voice_audio_data,
        warning_text,
    )


with gr.Blocks(title="Story Lab") as demo:
    gr.Markdown(
        "## Story Lab\nStart from a seed idea, explore a beat, iterate on dialogue, style, and visuals."
    )

    with gr.Row():
        with gr.Column():
            seed_prompt = gr.Textbox(
                label="Seed Concept",
                value="A shy linguistics student discovers a dead language can summon storms.",
                lines=3,
                placeholder="High-level pitch or logline",
            )
            beat_index = gr.Slider(
                label="Beat to Explore",
                value=1,
                minimum=1,
                maximum=5,
                step=1,
            )
            target_length = gr.Slider(
                label="Target Scene Length (words)",
                value=180,
                minimum=80,
                maximum=600,
                step=10,
            )
            style_directive = gr.Textbox(
                label="Style Transfer Directive",
                value="magical realism with lightly lyrical cadence, present tense",
                lines=2,
            )
            enable_images = gr.Checkbox(label="Generate Visual Concepts", value=True)
            image_seed = gr.Textbox(
                label="Image Seed (optional)",
                placeholder="Leave blank for random seed",
            )
        with gr.Column():
            char_table = gr.Dataframe(
                headers=CHAR_HEADERS,
                value=DEFAULT_CHARACTERS,
                datatype=["str", "str", "str", "str"],
                row_count=(2, "dynamic"),
                col_count=(4, "fixed"),
                label="Dialogue Roster",
            )
            scene_goal = gr.Textbox(
                label="Dialogue Scene Goal",
                value="Negotiate boundaries for trying the chant on the pier.",
            )
            conflict_axis = gr.Textbox(
                label="Primary Conflict Axis",
                value="curiosity vs caution",
            )
            turns = gr.Slider(
                label="Dialogue Turns",
                value=8,
                minimum=2,
                maximum=16,
                step=1,
            )
            portrait_focus = gr.Textbox(
                label="Portrait Focus",
                value=DEFAULT_CHARACTERS[0][0],
                placeholder="Character name to visualise",
            )
            portrait_wardrobe = gr.Textbox(
                label="Portrait Wardrobe Hint",
                value="storm scholar attire with weathered cloak",
                placeholder="Optional costume or styling cues",
            )
            generate_btn = gr.Button("Run Story Pipeline", variant="primary")

    with gr.Tabs():
        with gr.TabItem("Narrative"):
            brief_out = gr.Markdown(label="Expanded Brief")
            scene_json_out = gr.JSON(label="Scene Plan")
            prose_out = gr.Textbox(label="Generated Scene", lines=14)
            dialogue_out = gr.Textbox(label="Dialogue Draft", lines=12)
            styled_out = gr.Textbox(label="Styled Rewrite", lines=18)
        with gr.TabItem("Visuals"):
            scene_prompt_out = gr.Textbox(label="Scene Prompt", lines=10)
            scene_image_out = gr.Image(label="Scene Concept", type="pil")
            portrait_prompt_out = gr.Textbox(label="Portrait Prompt", lines=8)
            portrait_image_out = gr.Image(label="Character Portrait", type="pil")
            warning_out = gr.Markdown(label="Diagnostics")
        with gr.TabItem("Voice"):
            voice_summary_out = gr.Markdown(label="Voice Summary")
            voice_segments_out = gr.Dataframe(
                headers=VOICE_SEGMENT_HEADERS,
                datatype=["str"] * len(VOICE_SEGMENT_HEADERS),
                row_count=(1, "dynamic"),
                interactive=False,
                label="Segment Plan",
            )
            voice_ssml_out = gr.Textbox(label="SSML Plan", lines=10)
            voice_script_out = gr.Textbox(label="Narration Script", lines=10)
            voice_audio_out = gr.Audio(label="Narration Preview", type="numpy")

    generate_btn.click(
        fn=generate_story_bundle,
        inputs=[
            seed_prompt,
            beat_index,
            target_length,
            char_table,
            scene_goal,
            conflict_axis,
            turns,
            style_directive,
            enable_images,
            image_seed,
            portrait_focus,
            portrait_wardrobe,
        ],
        outputs=[
            brief_out,
            scene_json_out,
            prose_out,
            dialogue_out,
            styled_out,
            scene_prompt_out,
            scene_image_out,
            portrait_prompt_out,
            portrait_image_out,
            voice_summary_out,
            voice_segments_out,
            voice_ssml_out,
            voice_script_out,
            voice_audio_out,
            warning_out,
        ],
    )

if __name__ == "__main__":
    demo.launch()
