"""Load the embedding model and expose a single embed(text=..., images=...) call."""

import mlx.core as mx
from mlx_vlm import load


def load_embedder():
    model, processor = load("mlx-community/embeddinggemma-2-bf16")

    def embed(text=None, images=None) -> mx.array:
        inputs = processor(text=text, images=images, return_tensors="np")
        inputs = {key: mx.array(value) for key, value in inputs.items()}
        return model(**inputs).text_embeds

    return embed
