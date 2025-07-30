import os, random, torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = int(os.environ.get("MMSTORY_SEED", "42"))
random.seed(SEED); torch.manual_seed(SEED)

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose a default; allow override via env
MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
