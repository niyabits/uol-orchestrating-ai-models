from typing import List, Dict
from .prompts import DIALOGUE_SYSTEM
from .decoding import chat_generate, DecodeCfg


def generate_dialogue(characters: List[Dict[str, str]],
                      scene_goal: str,
                      conflict_axis: str,
                      turns: int = 10) -> str:
    roster = "\n".join(f"- {c['name']} ({c['role']}), objective: {c['objective']}, emotion: {c['emotion']}" for c in characters)
    user = f"""Characters:
{roster}

Scene goal: {scene_goal}
Primary conflict axis: {conflict_axis}
Turn budget: {turns}

Write dialogue now."""
    prompt = f"<|system|>\n{DIALOGUE_SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"
    return chat_generate(prompt, DecodeCfg(max_new_tokens=500, temperature=0.9, top_p=0.92, repetition_penalty=1.06))
