from .prompts import CONTENT_PLAN_SYSTEM,STYLE_TRANSFER_SYSTEM, END
from .utils import extract_first_json_obj
from .decoding import chat_generate, DecodeCfg
from typing import Dict, Any
import json

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