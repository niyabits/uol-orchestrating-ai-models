#!/usr/bin/env python
# coding: utf-8

# # Story Generation

get_ipython().system('pip install -U transformers torch accelerate sentencepiece pydantic==2.\\* orjson --quiet')


import os, json, re, math, textwrap, orjson, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, StoppingCriteriaList


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE=="cuda" and torch.cuda.is_bf16_supported() else torch.float16

# Choose a small, open, instruction-tuned model that runs on CPU/GPU:
# Good starters: "microsoft/Phi-3-mini-4k-instruct" (2.7B), "Qwen/Qwen2.5-3B-Instruct"
# Heavier (needs good GPU/RAM): "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    device_map="auto" if DEVICE=="cuda" else None
).to(DEVICE)

streamer = TextStreamer(tokenizer, skip_special_tokens=True)
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)


# ## A disciplined generate() helper
# 
# - Why: Stable decoding & repeatable experiments.
# - Adds anti-repetition and length controls.

@dataclass
class DecodeCfg:
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.05
    stop: Optional[List[str]] = None   # e.g. ["</json>"]

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strs: List[str], tokenizer, start_idx: int):
        self.stop_strs = stop_strs
        self.tokenizer = tokenizer
        self.start_idx = start_idx  # length of the prompt tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the generated continuation, not the prompt
        gen_ids = input_ids[0, self.start_idx:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        for s in self.stop_strs:
            if s in text:
                return True
        return False

def chat_generate(messages_or_text, cfg: DecodeCfg = DecodeCfg()) -> str:
    # 1) Prepare a single text string from either chat messages or raw text
    if isinstance(messages_or_text, str):
        text = messages_or_text
    else:
        # messages_or_text = [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
        text = tokenizer.apply_chat_template(
            messages_or_text,
            tokenize=False,              # get a string, not tensors
            add_generation_prompt=True
        )

    # 2) Tokenize to get BOTH input_ids and attention_mask
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    input_len = input_ids.shape[-1]
    
    stopping_criteria = None
    if cfg.stop:
        stopping_criteria = StoppingCriteriaList([
            StopOnStrings(cfg.stop, tokenizer, start_idx=input_len)
        ])

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,          # <<<<<< important
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,    # safe default for many chat LMs
        )

    # Decode only the generated continuation
    gen_only = out_ids[0, input_len:]
    text_out = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    if cfg.stop:
        for s in cfg.stop:
            if s in text_out:
                text_out = text_out.split(s, 1)[0]
                break

    return text_out


# ## Prompt Expansion
# 
# Goal: Take a terse logline and produce a rich, multi-angle expansion (themes, conflicts, beats, constraints). Few-shot prompt sets expectations and yields structured bullets.

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

def expand_prompt(seed_prompt: str) -> str:
    messages = [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {"role": "user", "content": f"Seed prompt: {seed_prompt}\n\nProduce the structured expansion now."}
    ]
    return chat_generate(messages, DecodeCfg(max_new_tokens=700, temperature=0.8, top_p=0.9))


# Example:
expanded = expand_prompt("A shy linguistics student discovers a dead language can summon storms.")
print(expanded)


# ## 3) Scene Description (schema → prose, with JSON validation)
# 
# Goal: Derive a scene graph (who/where/when/mood/visuals/sensory beats) then synthesize evocative prose. We first ask for strict JSON (schema below), then render prose from it. If the model returns invalid JSON, we repair it.

from pydantic import BaseModel, Field, ValidationError

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

SCHEMA_JSON = json.dumps(SceneSchema.model_json_schema(), indent=2)

SCHEMA_SYSTEM = f"""You output ONLY valid JSON matching this Pydantic schema:
{SCHEMA_JSON}
Do not include the schema, do not include any explanations, comments, or Markdown.
Return exactly one JSON object.
After the closing brace, write the token </json> and nothing else.
"""

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

# Example:
sc = scene_json_from_outline(expanded, beat_index=1)
print(sc)
print(render_scene_prose(sc))


# ## 4) Character Dialogue (role conditioning + turn budget + beats)
# 
# We build speaker profiles and constrain output to a screenplay-like format with a turn budget and inline subtext cues in stage directions (kept short).

DIALOGUE_SYSTEM = """You write snappy, character-driven dialogue.
Output format:
SPEAKER: line
  (stage direction / subtext)
No narration; only dialogue and concise stage directions.
Honor each character's objective and emotion. Keep lines short (≤18 words).
"""

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

# Example:
chars = [
     {"name":"Mira","role":"protagonist","objective":"convince Arjun to leave","emotion":"wary"},
     {"name":"Arjun","role":"foil","objective":"stall for time","emotion":"deflective"},
]
dlg = generate_dialogue(chars, "Mira tries to get Arjun out before the storm hits.", "trust vs control", turns=8)
print(dlg)


# ## 5) Style Transfer (tone) with a two-pass content-preservation plan
# 
# Single-shot “rewrite in style X” often drifts. We mitigate with content planning:
# 
# Extract a content plan (facts, plot beats, entities) in JSON.
# 
# Rewrite to a target style/tone while anchoring to the plan.

END = "<|END_JSON|>"
CONTENT_PLAN_SYSTEM = f"""Extract a content plan JSON with keys:
- entities: [{{name, type, attributes?}}]
- events: [{{order, summary}}]
- constraints: [{{kind, text}}]  # e.g., must-keep metaphors, lexical items
Output only JSON (no markdown). After the closing brace, write {END} and nothing else.
"""

def extract_first_json_obj(s: str) -> str:
    s = re.sub(r"```(?:json)?|```", "", s, flags=re.IGNORECASE)
    start = s.find("{")
    if start == -1:
        raise RuntimeError("No opening brace found.")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    raise RuntimeError("Unbalanced braces; JSON not closed.")


def extract_content_plan(text: str) -> Dict[str, Any]:
    user = f"Extract the content plan from this passage:\n{text}\n"
    prompt = f"<|system|>\n{CONTENT_PLAN_SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"

    raw = chat_generate(
        prompt,
        DecodeCfg(
            max_new_tokens=500,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.01,
            stop=[END]   # requires the StopOnStrings fix you added earlier (check only generated text)
        )
    )

    # Trim at sentinel if the model included it before stop fired
    raw = raw.split(END, 1)[0]
    json_text = extract_first_json_obj(raw)  # robust extractor from (B)
    return json.loads(json_text)

STYLE_TRANSFER_SYSTEM = """You perform style transfer while preserving content.
Rules:
- Faithfully preserve entities and event order from the plan.
- Apply the requested tone/style features.
- Avoid archaic words unless asked.
- Keep output length within ±15% of the input length unless asked.
"""

def style_transfer(text: str, target_style: str) -> str:
    plan = extract_content_plan(text)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    user = f"""Target style/tone: {target_style}

Content plan (must preserve):
{plan_json}

Source passage:
{text}

Rewrite now in the target style while preserving facts and ordering."""
    prompt = f"<|system|>\n{STYLE_TRANSFER_SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"
    return chat_generate(prompt, DecodeCfg(max_new_tokens=700, temperature=0.7, top_p=0.9, repetition_penalty=1.02))

# Example:
src = "Mira stepped onto the flooded platform, the loudspeakers coughing warnings as wind bent the halyard."
print(style_transfer(src, "noir, clipped sentences, wry subtext, contemporary diction"))


# # Final

seed = "A shy linguistics student discovers a dead language can summon storms."
expanded = expand_prompt(seed)
print("=== EXPANDED PROMPT ===\n", expanded[:1200], "\n")

scene_plan = scene_json_from_outline(expanded, beat_index=1)
print("=== SCENE JSON ===")
print(json.dumps(scene_plan.model_dump(), ensure_ascii=False, indent=2))

scene_text = render_scene_prose(scene_plan, target_len=180)
print("\n=== SCENE PROSE ===\n", scene_text, "\n")

chars = [
    {"name":"Mira","role":"protagonist","objective":"test the storm-chant safely","emotion":"guarded"},
    {"name":"Arjun","role":"mentor","objective":"discourage reckless use","emotion":"anxious"}
]
dialogue = generate_dialogue(chars,
                             scene_goal="Negotiate boundaries for trying the chant on the pier.",
                             conflict_axis="curiosity vs caution",
                             turns=8)
print("=== DIALOGUE ===\n", dialogue, "\n")

styled = style_transfer(scene_text, "magical realism with lightly lyrical cadence, restrained metaphors, present tense")
print("=== STYLE-TRANSFERRED ===\n", styled)


# ## Notes, tips, and swaps
# 
# ### Model swaps:
# 
# Small & easy: microsoft/Phi-3-mini-4k-instruct, Qwen/Qwen2.5-3B-Instruct.
# 
# Mid/heavier: meta-llama/Meta-Llama-3.1-8B-Instruct.
# 
# If you prefer GGUF models via llama-cpp-python, load with:



