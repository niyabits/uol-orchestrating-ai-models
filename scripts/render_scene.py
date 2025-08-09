# scripts/render_scene.py
from pathlib import Path

import orjson

from mmstory.image import CharacterPortraitGenerator, SceneToImageGenerator
from mmstory.schema import SceneSchema


BUNDLE_DIR = Path("artifacts")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path("tests/data/scene_plan.json").read_bytes()
scene = SceneSchema.model_validate(orjson.loads(DATA))

scene_gen = SceneToImageGenerator()
scene_result = scene_gen.generate(scene, seed=1234)
scene_result.save(str(BUNDLE_DIR / "scene.png"))

char_gen = CharacterPortraitGenerator()
portrait_result = char_gen.generate(scene.characters[0], scene=scene, seed=1234)
portrait_result.save(str(BUNDLE_DIR / "portrait.png"))
