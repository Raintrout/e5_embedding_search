from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

EMBEDDING_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingPipeline:
    def __init__(self, model_name='intfloat/e5-large', device=EMBEDDING_DEVICE) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def embedding(self, input: List[str]):
        # Each input text should start with "query: " or "passage: ".
        # For tasks other than retrieval, you can simply use the "query: " prefix.
        
        # Tokenize the input texts
        batch_dict = self.tokenizer(
                input,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # Optionally Normalize Embeddings
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings