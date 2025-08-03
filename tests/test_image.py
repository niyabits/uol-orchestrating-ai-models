from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import orjson

from mmstory.image import (
    CharacterPortraitGenerator,
    ImageGenerationResult,
    SceneToImageGenerator,
)
from mmstory.schema import SceneSchema

DATA_DIR = Path(__file__).parent / "data"


class DummyBackend:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
    ) -> ImageGenerationResult:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        self.calls.append(payload)
        return ImageGenerationResult(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            image_bytes=b"dummy-png",
            metadata={"backend": "dummy"},
        )


def _load_scene() -> SceneSchema:
    raw = (DATA_DIR / "scene_plan.json").read_bytes()
    return SceneSchema.model_validate(orjson.loads(raw))


def test_scene_to_image_prompt_longform() -> None:
    scene = _load_scene()
    backend = DummyBackend()
    generator = SceneToImageGenerator(backend=backend)

    result = generator.generate(scene)

    text = result.prompt.lower()
    assert "oxford university" in text
    assert "mysterious and suspenseful" in text
    assert "ancient manuscript" in text
    assert backend.calls, "backend should be invoked"
    call = backend.calls[0]
    assert call["width"] == 1024
    assert call["height"] == 576
    assert call["guidance_scale"] > 0
    assert result.metadata["backend"] == "dummy"


def test_character_portrait_prompt_links_scene_context() -> None:
    scene = _load_scene()
    character = scene.characters[0]
    backend = DummyBackend()
    generator = CharacterPortraitGenerator(backend=backend)

    result = generator.generate(character, scene=scene, mood_override="resolute focus")

    text = result.prompt.lower()
    assert character.name.lower() in text
    assert "resolute focus" in text
    assert scene.location.lower() in text
    assert backend.calls, "portrait backend should receive one call"
    call = backend.calls[0]
    assert call["height"] == 1024
    assert call["width"] == 768
    assert call["negative_prompt"] == generator.negative_prompt
