"""RAG pipeline: retrieve relevant chunks from Chroma, then ask Claude to
answer using only that retrieved context.

Run with: python src/rag.py "your question here"
"""
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
GROQ_MODEL = "qwen/qwen3.8-27b"
TOP_K = 5

_embedder = None
_collection = None
_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = chroma_client.get_or_create_collection("vnstock_docs")
    return _collection


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    embedder = get_embedder()
    query_embedding = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    results = get_collection().query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['metadata']['symbol']} / {c['metadata']['source']}]\n{c['text']}"
        for c in chunks
    )
    return (
        "Use the following context about Vietnamese companies to answer the question. "
        "If the context doesn't contain the answer, say so honestly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def answer_question(question: str) -> dict:
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)

    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [c["metadata"] for c in chunks],
    }


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Ask a question: ")

    result = answer_question(question)
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['symbol']} ({s['source']})")


if __name__ == "__main__":
    main()
