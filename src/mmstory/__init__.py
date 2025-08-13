from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "expand_prompt",
    "scene_json_from_outline",
    "render_scene_prose",
    "generate_dialogue",
    "style_transfer",
    "SceneToImageGenerator",
    "CharacterPortraitGenerator",
    "ImageGenerationResult",
    "DiffusersImageBackend",
    "SceneSchema",
    "Character",
    "SensoryBeat",
    "DecodeCfg",
    "TextSentimentClassifier",
    "EmotionTagger",
    "SentimentResult",
    "EmotionResult",
    "VoiceSynthesisModel",
    "VoiceSynthesisResult",
    "VoiceSegment",
]

_ATTR_TO_MODULE = {
    "expand_prompt": (".expand", "expand_prompt"),
    "scene_json_from_outline": (".scene", "scene_json_from_outline"),
    "render_scene_prose": (".scene", "render_scene_prose"),
    "generate_dialogue": (".dialogue", "generate_dialogue"),
    "style_transfer": (".style", "style_transfer"),
    "SceneToImageGenerator": (".image", "SceneToImageGenerator"),
    "CharacterPortraitGenerator": (".image", "CharacterPortraitGenerator"),
    "ImageGenerationResult": (".image", "ImageGenerationResult"),
    "DiffusersImageBackend": (".image", "DiffusersImageBackend"),
    "SceneSchema": (".schema", "SceneSchema"),
    "Character": (".schema", "Character"),
    "SensoryBeat": (".schema", "SensoryBeat"),
    "DecodeCfg": (".decoding", "DecodeCfg"),
    "TextSentimentClassifier": (".affect", "TextSentimentClassifier"),
    "EmotionTagger": (".affect", "EmotionTagger"),
    "SentimentResult": (".affect", "SentimentResult"),
    "EmotionResult": (".affect", "EmotionResult"),
    "VoiceSynthesisModel": (".voice", "VoiceSynthesisModel"),
    "VoiceSynthesisResult": (".voice", "VoiceSynthesisResult"),
    "VoiceSegment": (".voice", "VoiceSegment"),
}


def __getattr__(name: str) -> Any:
    if name not in _ATTR_TO_MODULE:
        raise AttributeError(f"module 'mmstory' has no attribute {name!r}")
    module_name, attr_name = _ATTR_TO_MODULE[name]
    module = import_module(module_name, __name__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .expand import expand_prompt
    from .scene import scene_json_from_outline, render_scene_prose
    from .dialogue import generate_dialogue
    from .style import style_transfer
    from .image import (
        SceneToImageGenerator,
        CharacterPortraitGenerator,
        ImageGenerationResult,
        DiffusersImageBackend,
    )
    from .schema import SceneSchema, Character, SensoryBeat
    from .decoding import DecodeCfg
    from .affect import (
        TextSentimentClassifier,
        EmotionTagger,
        SentimentResult,
        EmotionResult,
    )
    from .voice import VoiceSynthesisModel, VoiceSynthesisResult, VoiceSegment
