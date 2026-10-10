import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def load_images(directory: Path) -> tuple[list[Path], list[Image.Image]]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    images = [Image.open(path).convert("RGB") for path in paths]
    return paths, images


def main():
    parser = argparse.ArgumentParser(
        description="Embed a question and rank images by similarity."
    )
    parser.add_argument("question", help="The question (search query).")
    parser.add_argument("directory", type=Path, help="Directory containing images.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")

    paths, images = load_images(args.directory)
    if not paths:
        print(f"No image files found in {args.directory}", file=sys.stderr)
        return

    model, processor = load("mlx-community/embeddinggemma-2-bf16")

    def embed(text=None, images=None) -> mx.array:
        inputs = processor(text=text, images=images, return_tensors="np")
        inputs = {key: mx.array(value) for key, value in inputs.items()}
        return model(**inputs).text_embeds

    query_embedding = embed(text=[f"task: search result | query: {args.question}"])
    image_embeddings = embed(images=[[image] for image in images])

    scores = (query_embedding @ image_embeddings.T)[0]
    for idx in mx.argsort(scores).tolist()[::-1]:
        print(f"{scores[idx].item():.4f} | {paths[idx]}")


if __name__ == "__main__":
    main()
