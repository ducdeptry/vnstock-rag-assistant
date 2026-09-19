# Vnstock RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about major Vietnamese companies, grounded in real financial data pulled via [`vnstock`](https://github.com/thinh-vu/vnstock).

> ✅ Status: core pipeline working end-to-end (retrieval + generation + demo UI). Evaluation and retrieval-quality tuning are next.

## Why this project

Most RAG demos are generic "chat with a PDF" clones over English text. This one is grounded in Vietnamese-language financial data, which surfaces a real design decision most tutorials skip: the embedding model has to actually understand Vietnamese. Using a multilingual embedding model (see below) means an **English question correctly retrieves the relevant Vietnamese source text** — verified during development, not assumed.

## Tech stack

- **Data source:** [`vnstock`](https://github.com/thinh-vu/vnstock) — Vietnamese stock market data (company overviews, history, news)
- **Embeddings:** `sentence-transformers` with `intfloat/multilingual-e5-small` — chosen specifically for Vietnamese-language support, unlike the English-only models most RAG tutorials default to
- **Vector store:** `chromadb` (local, persisted to `chroma_db/`)
- **LLM:** Groq API (`qwen/qwen3.8-27b`) — fast inference, free tier, and Qwen's multilingual strength suits the Vietnamese source data
- **Demo UI:** `streamlit`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Groq API key (https://console.groq.com/keys)
```

## Usage

Run these in order — each step's output feeds the next one:

```bash
python src/fetch_data.py    # pull company data from vnstock into data/raw/
python src/build_index.py   # chunk, embed, and store it in a local vector DB
streamlit run app.py        # launch the interactive demo
```

Or query it directly from the command line:
```bash
python src/rag.py "What does FPT company do?"
```

## Roadmap

- [x] Project scaffolding
- [x] Pull and store financial documents with `vnstock`
- [x] Chunk + embed documents into a vector store
- [x] Basic retrieve-then-generate pipeline
- [x] Streamlit demo UI
- [ ] Evaluation set + retrieval/answer quality metrics
- [ ] Compare chunking/retrieval strategies
- [ ] Write-up of findings in this README

## License

MIT
