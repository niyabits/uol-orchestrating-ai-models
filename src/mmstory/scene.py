import re, json, orjson
from .prompts import SCHEMA_SYSTEM
from .schema import SceneSchema 
from .decoding import chat_generate, DecodeCfg

def scene_json_from_outline(expanded_outline_md: str, beat_index: int = 1) -> SceneSchema:
    user = f"""Given this expanded outline (Markdown), select Beat #{beat_index} and produce a scene JSON instance."""
    # Put the outline in the same user turn, but AFTER a clear delimiter:
    user += f"\n\n--- EXPANDED OUTLINE START ---\n{expanded_outline_md}\n--- EXPANDED OUTLINE END ---\n"

    prompt = f"<|system|>\n{SCHEMA_SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"

    raw = chat_generate(
        prompt,
        DecodeCfg(
            max_new_tokens=700,
            temperature=0.4,
            top_p=0.95,
            repetition_penalty=1.01,
            stop=["</json>"]  # << sentinel
        )
    )
    
    # raw should now be *just* JSON (no prompt), without the sentinel.
    text = raw.strip()

    # Safety: strip accidental code fences if any model tries to add them
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE|re.MULTILINE).strip()

    try:
        data = orjson.loads(text)
        return SceneSchema.model_validate(data)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Model did not return JSON.\n---\n{text[:800]}")
        data = orjson.loads(m.group(0))
        return SceneSchema.model_validate(data)

def render_scene_prose(scene: SceneSchema, target_len: int = 180) -> str:
    guide = f"""Write ~{target_len} words of vivid third-person limited prose.
Keep internal state consistent with 'pov'. Use camera_style as inspiration for sentence rhythm and framing.
Weave in at least 2 sensory beats. Avoid cliché.
"""
    content_plan = json.dumps(scene.model_dump(), ensure_ascii=False, indent=2)
    prompt = f"<|system|>\nYou turn structured scene plans into concise, evocative prose.\n<|user|>\nScene Plan (JSON):\n{content_plan}\n\nInstructions:\n{guide}\n<|assistant|>\n"
    return chat_generate(prompt, DecodeCfg(max_new_tokens=400, temperature=0.85, top_p=0.9, repetition_penalty=1.03))
