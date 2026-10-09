import argparse
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
TEXT_EXTENSIONS = {".txt"}


def build_documents(directory: Path) -> list[dict]:
    documents = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            documents.append({"text": path.read_text(encoding="utf-8")})
        elif suffix in IMAGE_EXTENSIONS:
            documents.append({"image": str(path)})
    return documents


def display(document: dict) -> str:
    if "text" in document:
        return document["text"]
    if "image" in document:
        return document["image"]
    return str(document)


def main():
    parser = argparse.ArgumentParser(description="Embed a query against a directory of files.")
    parser.add_argument("query", help="The search query.")
    parser.add_argument("directory", type=Path, help="Directory containing .txt and image files.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    model = SentenceTransformer(
        "unsloth/embeddinggemma-2",
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},
    )

    documents = build_documents(args.directory)
    if not documents:
        print(f"No .txt or image files found in {args.directory}", file=sys.stderr)
        return
    embedding_dim = 768

    query_embedding = model.encode(
        [args.query],
        prompt_name="SearchQuery",
        truncate_dim=embedding_dim,
        normalize_embeddings=True,
    )
    document_embeddings = model.encode(
        documents,
        prompt_name="Document",
        batch_size = 4,
        truncate_dim=embedding_dim,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    scores = model.similarity(query_embedding, document_embeddings)[0]
    for idx in scores.argsort(descending=True).tolist():
        print(f"{scores[idx].item():.4f} | {display(documents[idx])}")


if __name__ == "__main__":
    main()
