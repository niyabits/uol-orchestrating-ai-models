from pydantic import BaseModel, Field, ValidationError
from typing import List

class Character(BaseModel):
    name: str
    role: str
    objective: str
    emotion: str

class SensoryBeat(BaseModel):
    modality: str  # e.g., "visual", "auditory", "tactile", "olfactory"
    detail: str

class SceneSchema(BaseModel):
    location: str
    time: str
    weather: str
    mood: str
    pov: str
    camera_style: str
    characters: List[Character]
    key_props: List[str] = Field(default_factory=list)
    beats: List[str]
    sensory: List[SensoryBeat]