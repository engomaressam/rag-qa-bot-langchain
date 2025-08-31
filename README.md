# RAG QA bot: ask questions about a PDF with LangChain and local models

A small retrieval-augmented generation (RAG) app: upload a PDF in a Gradio page, ask a question, and get an answer grounded in the most relevant passages of that document. Everything runs locally; no cloud API key is needed.

Built for IBM's *Project: Generative AI Applications with RAG and LangChain* (IBM AI Engineering Professional Certificate). The course labs use IBM watsonx; this version swaps in open models so it runs on any machine.

## The pipeline (`main_qa_bot.py`)

| Step | Implementation |
|---|---|
| Load | `PyPDFLoader` reads the uploaded PDF page by page |
| Split | `RecursiveCharacterTextSplitter`, 1,000-character chunks with 200 characters of overlap |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings` (GPU if available, else CPU) |
| Store | an in-memory Chroma vector store built per document |
| Retrieve | the 4 most similar chunks |
| Answer | a LangChain `RetrievalQA` chain ("stuff": the 4 chunks go into one prompt) |
| Model | **Llama 2 through Ollama** when Ollama is installed; otherwise Hugging Face's `microsoft/DialoGPT-medium` |
| UI | Gradio: a file box, a question box and the answer, on http://localhost:7860 |

Use the Ollama path. DialoGPT is a conversational model, not an instruction-following one, so the fallback keeps the app running but gives weak answers; an instruction-tuned model (Llama 2, Mistral, FLAN-T5) is what makes the answers useful.

## The course labs, rewritten for local models

| Folder | What the scripts cover |
|---|---|
| `01-loading-and-splitting/` | document loaders for text, PDF, Markdown, JSON, CSV, Word and web pages; character, recursive, Markdown-header and code-aware (Python, JavaScript, LaTeX) splitters; why stuffing a whole document into a prompt fails |
| `02-embeddings-and-retrieval/` | embeddings with sentence-transformers (and watsonx, the course original); Chroma and FAISS vector stores; similarity, MMR, multi-query, self-query and parent-document retrievers |

The `*_local.py` scripts run anywhere. The scripts without `_local` are the watsonx versions and need the IBM Skills Network lab environment. The sample documents (IBM's lab texts and PDFs) are downloaded from IBM's public URLs at run time.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # or: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama2                                   # optional but recommended (https://ollama.com)
python main_qa_bot.py                                # http://localhost:7860
```

The first run downloads the embedding model (about 90 MB) and, without Ollama, the DialoGPT model. `start_qa_bot.bat` does the install and launch in one step on Windows.

## Limits

There are no automated tests and no evaluation of answer quality on a question set. The vector store is rebuilt for every question, which is fine for one document at a time but slow for large PDFs; caching the store per file would be the first improvement.

## History

Built in August 2025 on another GitHub account and moved here in October 2026. Only my code moved: IBM's lab instructions, notebooks and the grading screenshots stayed behind.

---

Omar Essam El Sherbiny · [omaressam.web.app](https://omaressam.web.app) · [LinkedIn](https://www.linkedin.com/in/omar-essam-el-sherbiny-b602279b/)
