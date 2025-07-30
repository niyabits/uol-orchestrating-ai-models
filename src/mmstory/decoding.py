from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from .llm import get_model, get_tokenizer
from .config import DEVICE

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
    tokenizer = get_tokenizer()
    model = get_model()
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