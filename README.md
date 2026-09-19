# Vnstock RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about Vietnamese stocks and companies, grounded in real financial data pulled via [`vnstock`](https://github.com/thinh-vu/vnstock).

> 🚧 Status: early development — see roadmap below.

## Why this project

Most RAG demos are generic "chat with a PDF" clones over English text. This one is grounded in Vietnamese-language financial data, and focuses on measuring and improving retrieval quality rather than just wiring an LLM to a vector store.

## Tech stack

- **Data source:** [`vnstock`](https://github.com/thinh-vu/vnstock) — Vietnamese stock market data
- **Embeddings:** `sentence-transformers` (local, free)
- **Vector store:** `chromadb`
- **LLM:** Claude (via the `anthropic` SDK)
- **Demo UI:** `streamlit`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Anthropic API key
```

## Roadmap

- [x] Project scaffolding
- [ ] Pull and store financial documents with `vnstock`
- [ ] Chunk + embed documents into a vector store
- [ ] Basic retrieve-then-generate pipeline
- [ ] Streamlit demo UI
- [ ] Evaluation set + retrieval/answer quality metrics
- [ ] Compare chunking/retrieval strategies
- [ ] Write-up of findings in this README

## License

MIT
