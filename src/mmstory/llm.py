from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODEL_NAME, DEVICE, DTYPE

_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None
        ).to(DEVICE)
    return _model
