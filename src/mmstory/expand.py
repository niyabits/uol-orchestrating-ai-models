from .decoding import chat_generate, DecodeCfg
from .prompts import EXPANSION_SYSTEM

def expand_prompt(seed_prompt: str) -> str:
    messages = [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {"role": "user", "content": f"Seed prompt: {seed_prompt}\n\nProduce the structured expansion now."}
    ]
    return chat_generate(messages, DecodeCfg(max_new_tokens=700, temperature=0.8, top_p=0.9))