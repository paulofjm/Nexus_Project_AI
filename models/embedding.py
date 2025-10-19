from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_embed_model.eval()

def embed_text(texts: list[str] | str) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    return _embed_model.encode(texts)
