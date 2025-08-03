"""Image generation utilities for transforming narrative structures into visual assets.

The module provides two high-level generators:

* :class:`SceneToImageGenerator` converts a structured :class:`~mmstory.schema.SceneSchema`
  instance into a cinematic prompt and drives a diffusion backend. FLUX.1 Schnell is used by
  default because it trades a small amount of photorealism for remarkable layout coherence and
  fast iteration speed – exactly what storyboard ideation calls for.
* :class:`CharacterPortraitGenerator` focuses on hero shots of individual characters, leaning on
  the SDXL Realistic Vision family to preserve facial structure while still responding strongly to
  textual style cues. The result is a portrait-driven prompt tuned for casting explorations.

Both generators share an ``ImageBackend`` protocol so the heavy-weight diffusion pipeline can be
swapped for deterministic stubs in unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import orjson

from .schema import Character, SceneSchema, SensoryBeat


@dataclass
class ImageGenerationResult:
    """Container for returning image bytes alongside the textual prompt."""

    prompt: str
    negative_prompt: Optional[str]
    width: int
    height: int
    seed: Optional[int]
    image_bytes: Optional[bytes]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Persist the generated PNG bytes to ``path``.

        This helper is intentionally light-weight so downstream notebooks can stash draft renders
        without needing to juggle PIL imports in every script.
        """

        if self.image_bytes is None:
            raise RuntimeError("No image bytes available to save.")
        with open(path, "wb") as fh:
            fh.write(self.image_bytes)


class ImageBackend(Protocol):
    """Interface all backends have to expose to the high-level generators."""

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
        ...


class DiffusersImageBackend:
    """Thin wrapper around ``diffusers`` text-to-image pipelines.

    The backend is lazy – we only instantiate the underlying pipeline when ``__call__`` is invoked,
    letting unit tests swap in stubs without importing heavyweight assets.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        torch_dtype: Any = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:  # pragma: no cover - exercised only when diffusers missing
            raise RuntimeError(
                "The diffusers package is required for DiffusersImageBackend"
            ) from exc

        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        )
        if self.device is not None:
            pipeline = pipeline.to(self.device)
        self._pipeline = pipeline
        return pipeline

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
        pipeline = self._ensure_pipeline()

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = output.images[0]
        meta = {
            "model": self.model_id,
            "safety_checker": getattr(output, "nsfw_content_detected", None),
        }
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return ImageGenerationResult(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            image_bytes=buffer.getvalue(),
            metadata=meta,
        )


def _coerce_scene(scene: SceneSchema | str | Dict[str, Any]) -> SceneSchema:
    if isinstance(scene, SceneSchema):
        return scene
    if isinstance(scene, dict):
        return SceneSchema.model_validate(scene)
    try:
        data = orjson.loads(scene)
    except orjson.JSONDecodeError:
        data = json.loads(scene)
    return SceneSchema.model_validate(data)


def _coerce_character(character: Character | Dict[str, Any]) -> Character:
    if isinstance(character, Character):
        return character
    return Character.model_validate(character)


def _format_sensory_details(sensory: Sequence[SensoryBeat]) -> str:
    if not sensory:
        return ""
    fragments = [f"{beat.modality.lower()}: {beat.detail}" for beat in sensory]
    return "; ".join(fragments)


class SceneToImageGenerator:
    """Compose cinematic prompts from :class:`SceneSchema` instances and render them.

    The default backend is :class:`DiffusersImageBackend` pointing at ``black-forest-labs/FLUX.1-schnell``.
    FLUX excels at translating long-form prompts into grounded spatial layouts, a property that makes
    it a strong default for storyboards where you care more about staging than photorealistic skin.
    """

    def __init__(
        self,
        *,
        backend: Optional[ImageBackend] = None,
        backend_factory: Optional[Callable[[], ImageBackend]] = None,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 28,
        guidance_scale: float = 6.5,
        negative_prompt: str = (
            "lowres, oversaturated, flat lighting, incoherent anatomy, text artifacts, watermark"
        ),
    ) -> None:
        self._backend = backend
        self._backend_factory = backend_factory
        if backend is None and backend_factory is None:
            self._backend_factory = lambda: DiffusersImageBackend(model_id=model_id)
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt

    def _ensure_backend(self) -> ImageBackend:
        if self._backend is None:
            if self._backend_factory is None:
                raise RuntimeError("No backend configured for SceneToImageGenerator")
            self._backend = self._backend_factory()
        return self._backend

    def build_prompt(self, scene: SceneSchema) -> str:
        characters = ", ".join(
            f"{char.name} ({char.role.lower()}, {char.emotion.lower()})"
            for char in scene.characters
        )
        key_props = ", ".join(scene.key_props)
        beats = " ".join(scene.beats)
        sensory = _format_sensory_details(scene.sensory)

        prompt_parts: List[str] = [
            f"Cinematic concept art of {scene.location} during {scene.time}",
            f"moody {scene.weather.lower()} ambience",
            f"overall tone {scene.mood.lower()}",
            f"camera style {scene.camera_style.lower()}",
        ]
        if characters:
            prompt_parts.append(f"featuring {characters}")
        if key_props:
            prompt_parts.append(f"iconic props: {key_props}")
        if beats:
            prompt_parts.append(f"story beats: {beats}")
        if sensory:
            prompt_parts.append(f"sensory cues {sensory}")
        prompt_parts.append("rendered in volumetric lighting, dramatic depth of field, 32mm lens")
        return ", ".join(part.strip() for part in prompt_parts if part.strip())

    def generate(
        self,
        scene: SceneSchema | str | Dict[str, Any],
        *,
        seed: Optional[int] = None,
    ) -> ImageGenerationResult:
        scene_schema = _coerce_scene(scene)
        prompt = self.build_prompt(scene_schema)
        backend = self._ensure_backend()
        return backend(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=seed,
        )


class CharacterPortraitGenerator:
    """Generate hero-shot portraits for :class:`Character` definitions.

    SDXL Realistic Vision prioritises facial detail while staying pliable to textual descriptions of
    attire and mood. That balance makes it suitable for casting exploration without fighting the
    model over stylisation.
    """

    def __init__(
        self,
        *,
        backend: Optional[ImageBackend] = None,
        backend_factory: Optional[Callable[[], ImageBackend]] = None,
        model_id: str = "SG161222/Realistic_Vision_V5.1-noVAE",
        width: int = 768,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        style_preset: str = "portrait photography, Rembrandt lighting, shallow depth of field",
        negative_prompt: str = (
            "low detail, deformed face, extra limbs, text watermark, exaggerated proportions"
        ),
    ) -> None:
        self._backend = backend
        self._backend_factory = backend_factory
        if backend is None and backend_factory is None:
            self._backend_factory = lambda: DiffusersImageBackend(model_id=model_id)
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.style_preset = style_preset
        self.negative_prompt = negative_prompt

    def _ensure_backend(self) -> ImageBackend:
        if self._backend is None:
            if self._backend_factory is None:
                raise RuntimeError("No backend configured for CharacterPortraitGenerator")
            self._backend = self._backend_factory()
        return self._backend

    def build_prompt(
        self,
        character: Character,
        *,
        scene: Optional[SceneSchema | Dict[str, Any] | str] = None,
        wardrobe: Optional[str] = None,
        mood_override: Optional[str] = None,
    ) -> str:
        mood = mood_override or character.emotion
        wardrobe_desc = wardrobe or f"attire inspired by {character.role.lower()} archetypes"
        prompt_parts = [
            f"Ultra-detailed portrait of {character.name}",
            f"expression {mood.lower()}",
            wardrobe_desc,
            f"motivated lighting hinting at {character.objective.lower()}",
            self.style_preset,
        ]
        if scene is not None:
            scene_schema = _coerce_scene(scene)
            prompt_parts.append(
                f"set within {scene_schema.location.lower()} during {scene_schema.time.lower()}"
            )
            prompt_parts.append(f"ambient mood {scene_schema.mood.lower()}")
        return ", ".join(part.strip() for part in prompt_parts if part.strip())

    def generate(
        self,
        character: Character | Dict[str, Any],
        *,
        scene: Optional[SceneSchema | Dict[str, Any] | str] = None,
        wardrobe: Optional[str] = None,
        mood_override: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ImageGenerationResult:
        char_schema = _coerce_character(character)
        prompt = self.build_prompt(
            char_schema,
            scene=scene,
            wardrobe=wardrobe,
            mood_override=mood_override,
        )
        backend = self._ensure_backend()
        return backend(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            seed=seed,
        )


__all__ = [
    "ImageGenerationResult",
    "ImageBackend",
    "DiffusersImageBackend",
    "SceneToImageGenerator",
    "CharacterPortraitGenerator",
]
