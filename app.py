"""Interactive Gradio UI for the mmstory toolchain.

This surfaces the same primitives used in the test smoke script, but in a
workflow that mirrors how an author or narrative designer might iterate on a
concept. Users can start from a seed idea, expand it into a brief, drill into a
specific beat, and quickly spin up dialogue and styled prose variants.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import gradio as gr

from mmstory import (
    expand_prompt,
    generate_dialogue,
    render_scene_prose,
    scene_json_from_outline,
    style_transfer,
)

CHAR_HEADERS = ["name", "role", "objective", "emotion"]
DEFAULT_CHARACTERS = [
    ["Elizabeth Hawthorne", "Protagonist", "Master the storm-binding dialect", "Determined"],
    ["Brother Ambrose", "Mentor", "Guide Elizabeth without breaking his vows", "Guarded"],
]


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


def generate_story_bundle(
    seed_prompt: str,
    beat_index: int,
    target_length: int,
    characters: Iterable[Any],
    scene_goal: str,
    conflict_axis: str,
    turns: int,
    style_directive: str,
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
    return brief, scene_json, prose, dialogue, styled


with gr.Blocks(title="Story Lab") as demo:
    gr.Markdown(
        "## Story Lab\nStart from a seed idea, explore a beat, and iterate on dialogue and style."
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
            generate_btn = gr.Button("Run Story Pipeline", variant="primary")

    styled_out = gr.Textbox(label="Styled Rewrite", lines=18)

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
        ],
        outputs=[
            styled_out,
        ],
    )

if __name__ == "__main__":
    demo.launch()
