"""Chunk the raw company JSON into text documents, embed them, and store
them in a local Chroma vector database.

Run with: python src/build_index.py
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Multilingual model: our data is in Vietnamese, so an English-only
# embedding model (the common tutorial default) would perform poorly here.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def build_documents() -> list[dict]:
    """Turn raw overview/news JSON files into (text, metadata) documents."""
    documents = []

    for overview_file in sorted(RAW_DIR.glob("*_overview.json")):
        symbol = overview_file.stem.replace("_overview", "")
        data = json.loads(overview_file.read_text())

        profile_text = (
            f"Company: {symbol}\n"
            f"CEO: {data.get('ceo_name', 'N/A')}\n"
            f"Founded: {data.get('founded_date', 'N/A')}\n"
            f"Exchange: {data.get('exchange', 'N/A')}\n"
            f"Employees: {data.get('number_of_employees', 'N/A')}\n\n"
            f"Business model:\n{data.get('business_model', '')}"
        )
        for i, chunk in enumerate(chunk_text(profile_text)):
            documents.append({
                "id": f"{symbol}-overview-{i}",
                "text": chunk,
                "metadata": {"symbol": symbol, "source": "overview"},
            })

        if data.get("history"):
            history_text = f"Company: {symbol}\nHistory:\n{data['history']}"
            for i, chunk in enumerate(chunk_text(history_text)):
                documents.append({
                    "id": f"{symbol}-history-{i}",
                    "text": chunk,
                    "metadata": {"symbol": symbol, "source": "history"},
                })

    for news_file in sorted(RAW_DIR.glob("*_news.json")):
        symbol = news_file.stem.replace("_news", "")
        articles = json.loads(news_file.read_text())

        for i, article in enumerate(articles):
            text = f"Company: {symbol}\nNews: {article.get('title', '')}\n{article.get('head', '')}"
            documents.append({
                "id": f"{symbol}-news-{i}",
                "text": text,
                "metadata": {
                    "symbol": symbol,
                    "source": "news",
                    "publish_time": str(article.get("publish_time", "")),
                    "url": article.get("url", ""),
                },
            })

    return documents


def main():
    documents = build_documents()
    print(f"Built {len(documents)} document chunks from raw data.")

    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, may take a minute)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # e5 models expect a "passage: " prefix on stored documents
    # (and a "query: " prefix on search queries later).
    texts = [f"passage: {doc['text']}" for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("vnstock_docs")

    collection.upsert(
        ids=[doc["id"] for doc in documents],
        embeddings=embeddings.tolist(),
        documents=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
    )

    print(f"Indexed {len(documents)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
