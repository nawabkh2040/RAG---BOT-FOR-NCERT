# CBSE Study Assistant — Retrieval & Embeddings

This repository builds embeddings from a local knowledgebase (PDFs/text), creates a FAISS index using a MiniLM embedding model, and provides retrieval + chat helpers.

## Requirements
- Python 3.9+
- Virtual environment recommended
- See `requiremnets.txt` for packages; important ones:
  - `sentence-transformers` (MiniLM)
  - `faiss-cpu` (vector index)
  - `pypdf` (PDF parsing)

Note: Installing `faiss-cpu` on Windows can sometimes fail via `pip`. If `pip install faiss-cpu` fails, try conda:

```bash
conda install -c pytorch faiss-cpu
```

## Quick Setup (Windows)

1. Create and activate a venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or: .venv\Scripts\activate  # cmd.exe
```

2. Upgrade pip and install requirements

```powershell
python -m pip install --upgrade pip
pip install -r requiremnets.txt
pip install tqdm
```

3. If `pypdf` or other extras are missing, install them:

```powershell
pip install pypdf
```

## Build embeddings

Run the build step to parse the `knowledgebase/` folder, chunk text, compute embeddings with MiniLM, and write a FAISS index + metadata:

```powershell
python knowledgebase/convertTOembeding.py build --kb knowledgebase --out knowledgebase/embeddings
```

This will create:
- `knowledgebase/embeddings/faiss.index`
- `knowledgebase/embeddings/metadata.json`

## Query / Test retrieval

Quick CLI query (uses the same MiniLM model to encode query):

```powershell
python knowledgebase/convertTOembeding.py query "What is photosynthesis?" --out knowledgebase/embeddings --top-k 5
# or using the retrieval helper
python retrieval.py "What is photosynthesis?" --out knowledgebase/embeddings --top-k 5
```

## Chat integration

- `chat.py` contains `ChatService` which will automatically try to use the FAISS retriever (if `retrieval.py` and index exist).
- To run the sample chat:

```powershell
python chat.py
```

If you want the LLM integration, set an environment variable for the Google/Gemini key (optional):

```powershell
setx GEMINI_API_KEY "your_api_key_here"
# or set GOOGLE_API_KEY
```

## Embedding / Retriever internals
- Chunking: `convertTOembeding.py` splits text into word chunks (default 200 words, 50 overlap).
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` by default.
- Index: FAISS `IndexFlatIP` with normalized vectors (cosine similarity via inner product).

## Troubleshooting
- If build reports `No text chunks found` — check that `knowledgebase/` contains readable PDFs or text files and that `pypdf` is installed.
- If `faiss` import fails on Windows, prefer conda install as above.
- If embeddings are slow, reduce batch size in `convertTOembeding.py` or use a different model.

## Next steps (suggestions)
- Add a small Flask/FastAPI endpoint in `app.py` to expose a chat API backed by `ChatService`.
- Add source citation formatting into prompts (so the model can reference chapter/file names).

If you want, I can add the Flask endpoint or update `app.py` to call `ChatService` directly.
