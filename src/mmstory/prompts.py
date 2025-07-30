import json
from .schema import SceneSchema 

EXPANSION_SYSTEM = """You are a narrative development assistant.
Expand terse story prompts into a structured creative brief with:
- Premise (1–2 sentences)
- World/Setting (specific time/place, social context)
- Themes (3 bullets)
- Protagonist & Goal (bio + objective)
- Antagonistic Force (person/system/internal)
- Stakes (why it matters)
- Constraints (tone, POV, target length)
- 5-Beat Outline (Beat #: heading + 1–2 lines)
Return clean Markdown with headings and bullets.
"""

SCHEMA_JSON = json.dumps(SceneSchema.model_json_schema(), indent=2)
SCHEMA_SYSTEM = f"""You output ONLY valid JSON matching this Pydantic schema:
{SCHEMA_JSON}
Do not include the schema, do not include any explanations, comments, or Markdown.
Return exactly one JSON object.
After the closing brace, write the token </json> and nothing else.
"""

DIALOGUE_SYSTEM = """You write snappy, character-driven dialogue.
Output format:
SPEAKER: line
  (stage direction / subtext)
No narration; only dialogue and concise stage directions.
Honor each character's objective and emotion. Keep lines short (≤18 words).
"""

END = "<|END_JSON|>"
CONTENT_PLAN_SYSTEM = f"""Extract a content plan JSON with keys:
- entities: [{{name, type, attributes?}}]
- events: [{{order, summary}}]
- constraints: [{{kind, text}}]  # e.g., must-keep metaphors, lexical items
Output only JSON (no markdown). After the closing brace, write {END} and nothing else.
"""

STYLE_TRANSFER_SYSTEM = """You perform style transfer while preserving content.
Rules:
- Faithfully preserve entities and event order from the plan.
- Apply the requested tone/style features.
- Avoid archaic words unless asked.
- Keep output length within ±15% of the input length unless asked.
"""