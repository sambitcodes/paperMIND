# 📚🔍 PaperMIND — Chat with Research Papers (RAG)

PaperMIND is a Streamlit app that lets you upload a research paper PDF, index it into a vector database, and then ask questions grounded in the document using Retrieval‑Augmented Generation (RAG).

**Live app:** https://paperminded.streamlit.app/  

---

## Why RAG (in plain terms)

LLMs are strong at generating text, but they do not automatically “know” your uploaded PDF. RAG fixes this by **retrieving relevant passages from your document at question time** and supplying them as context to the model so the answer can be grounded in the paper.

In PaperMIND, RAG is implemented as two phases:

- **Indexing phase (one-time per PDF / per collection):** PDF → extracted text → chunks → embeddings → stored in ChromaDB. 
- **Query phase (every question):** question → query embedding → retrieve Top‑K chunks → pass chunks as `context` to the selected model → render answer + sources.

---

## App features

- Upload & index PDFs with configurable chunk size and overlap.   
- Persistent vector storage using ChromaDB collections.   
- RAG toggle (grounded answers when enabled).   
- Multiple models via a single orchestrator: Groq chat models + extractive local models.   
- “View sources” output: document name, page number, section/subsection, chunk id.   
 
---

## Complete architecture (end-to-end)

### 1) PDF extraction (PDF → elements)
PaperMIND uses **Unstructured** to partition the PDF into structured elements (titles, narrative blocks, lists, tables, etc.).   

Code location:
- `src/pdf_processing/pdf_processor.py` → `process_pdf()` calls `partition_pdf(...)` and returns `(elements, metadata)`.   

### 2) Chunking (elements → chunks with metadata)
The chunker walks through extracted elements, tracks section/subsection based on headings/titles, and builds text chunks with overlap to reduce boundary loss.   

Each chunk is represented as a `Chunk` dataclass containing:
- `text`, `page_number`, `section`, `subsection`, `element_type`, `chunk_id`.   
Code location:
- `src/pdf_processing/chunker.py` → `SmartChunker.chunk_elements(...)`.   

### 3) Embeddings (chunks → vectors)
During indexing, PaperMIND converts each chunk into a vector embedding using the configured embedding model via `CachedEmbedder`.   
Code location:
- `src/indexing/index_manager.py`:
  - `chunk_texts = [chunk.text for chunk in chunks]`
  - `embeddings = self.embedder.embed_documents(chunk_texts)`   

### 4) Vector store creation (vectors + text + metadata → ChromaDB)
PaperMIND stores embeddings in **ChromaDB** using a persistent client.   

What gets stored per chunk:
- `ids` → chunk ids (e.g. `paper_chunk_12`)
- `documents` → chunk text
- `embeddings` → vectors
- `metadatas` → page number, section/subsection, document name, etc.   

Code location:
- `src/indexing/vector_store.py`:
  - `create_collection(name)` uses `get_or_create_collection(...)`.   
  - `add_documents(...)` calls `collection.add(ids=..., embeddings=..., documents=..., metadatas=...)`.   
- `src/indexing/index_manager.py` calls these steps in `index_pdf(...)`.   
### 5) Retrieval (question → Top‑K relevant chunks)
At question time, PaperMIND:
1) embeds the question,  
2) queries Chroma for Top‑K nearest neighbors,  
3) fetches the corresponding chunk texts,  
4) returns both the texts and their citations.   

Code location:
- `src/retrieval/retriever.py` → `RAGRetriever.retrieve(query)`:
  - `query_embedding = self.embedder.embed_query(query)`   
  - `doc_ids, metadatas, distances = self.vector_store.query(..., n_results=self.top_k)`   
  - `collection.get(ids=kept_ids)` to fetch full chunk texts.   
### 6) Generation (chunks → answer)
When **Use document context (RAG)** is enabled in the UI, the chat layer passes retrieved text via the `context=` argument to the orchestrator so both chat models and extractive models can use it correctly.   
Code location:
- `streamlit_app/chat_interface.py` (your chat UI) retrieves `context` and calls:
  - `state.llm_orchestrator.stream_response(..., context=context)`   
- `src/llm/orchestrator.py` routes to:
  - Groq chat models (streaming)
  - local extractive models (yield once)   
---

## Models (what “RAG + models” means here)

Your UI model dropdown is populated using `LLMOrchestrator.get_available_models()`.   
### Groq chat models (generative)
These generate answers using the prompt and optional context, and stream tokens into the UI.   
### Extractive models (grounded)
Extractive models require context; they do not “guess” without it. That’s why passing `context=context` is crucial.   
> Note: If you are currently using the version where `SciBERTModel` wraps `pipeline("question-answering")`, it is an extractive QA approach driven by a QA-tuned checkpoint.    
---

## Running locally (complete steps)

### 1) System prerequisites (important for PDFs)
PaperMIND uses Unstructured + `pdf2image` + OCR tooling. Your `requirements.txt` notes that **Poppler and Tesseract are system dependencies**, not pip packages.    
Install system deps (examples):
- **Ubuntu/Debian** (example): `sudo apt-get install poppler-utils tesseract-ocr`
- **macOS** (example): `brew install poppler tesseract`

(Exact install commands can vary by OS; the key point is that Poppler + Tesseract must exist for best PDF extraction/OCR support.)    
### 2) Clone and create a virtual env
```bash
git clone <YOUR_REPO_URL>
cd papermind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```


### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, ChromaDB, Transformers/Torch, Groq SDK, Unstructured, etc.    
### 4) Create a `.env` file (required)

PaperMIND requires a Groq API key because Groq chat models are called from the orchestrator.    
Create a `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If you also want to pull private Hugging Face models (optional), add a token:

```env
HF_TOKEN=your_huggingface_token_here
```

Important: current code explicitly reads `GROQ_API_KEY` via `os.getenv("GROQ_API_KEY")`.    
(Only add `HF_TOKEN` if your runtime or deployment needs it for private models; public models generally don’t require it.)

### 5) Run Streamlit

Run whichever Streamlit entry file your repo uses. If your entry is `app.py`:

```bash
streamlit run app.py
```

If your entry file differs, run that file instead.

### 6) Use the app locally

- Upload a PDF in the sidebar.    
- Click **Index document** (creates a Chroma collection + stores embeddings).    
- Keep **Use document context (RAG)** enabled.    
- Choose a model and ask questions; open **View sources** to see citations.    
---

## Deployment notes (Streamlit Cloud)

- Add `GROQ_API_KEY` to Streamlit Cloud **Secrets**. The orchestrator will fail fast if it is missing.    
- The app uses Chroma persistent storage via `PersistentClient(...)`. In hosted environments, persistence depends on the platform’s filesystem behavior.    
---

## “Complete extraction process” summary (quick checklist)

1) `PDFProcessor.process_pdf()` → PDF → elements.    
2) `SmartChunker.chunk_elements(...)` → elements → chunks with metadata.    
3) `CachedEmbedder.embed_documents(...)` → chunk text → embeddings.    
4) `VectorStoreManager.create_collection(...)` → create/load Chroma collection.    
5) `VectorStoreManager.add_documents(...)` → store ids + embeddings + documents + metadata.    
6) `RAGRetriever.retrieve(...)` → query embed → Chroma query → chunk texts + citations.    
7) `chat_interface.py` → join chunks → call orchestrator with `context=context`.    
---

## Future upgrades (practical, high-impact)

### Retrieval improvements

- **Re-ranking (Cross Encoder):** retrieve Top‑K (e.g., 20) with embeddings, then re-rank using a cross-encoder and keep the best 5 for context. This typically improves answer quality and reduces irrelevant context. (Hook point: after `vector_store.query(...)` in `RAGRetriever.retrieve(...)`.) 
- **MMR / diversity selection:** reduce redundancy by selecting chunks that are relevant but not duplicative (improves context packing). (Hook point: between retrieval and `context` join.)    
- **Hybrid search:** combine vector search with keyword search (BM25) for equations, citations, and exact terminology. (Hook point: retrieval layer.)    


### Safety \& Guardrails

- **Context-required guardrail for extractive models:** if `context` is empty, return a clear “not found in document” response instead of calling the model. (Hook point: `chat_interface.py` right after context building.)    
- **Similarity threshold filtering:** only allow chunks with distances/similarity within an acceptable range; otherwise refuse to answer from the doc. (Hook point: use `distances` returned by retriever.)     
- **Prompt-injection defense:** detect instructions inside retrieved text that attempt to override system behavior; filter or de-weight those chunks. (Hook point: before passing `context` into orchestrator.)    


### Product/UX upgrades

- **Better source rendering:** show chunk previews and distances next to citations for debugging and trust. (Hook point: `chat_interface.py` and retrieval result payload.)    
- **Multi-document collections:** index multiple PDFs into one collection and show per-document citations (your metadata already stores `document`).    
---

## Tech stack

- Streamlit UI    
- Unstructured for PDF extraction    
- ChromaDB vector store
- Transformers + Torch for NLP models 
- Groq SDK for hosted chat LLMs    

---

## Contact

For questions, issues, or collaboration:

- **Project:** PaperMIND
- **Live app:** [PaperMIND](https://paperminded.streamlit.app/)

Add your preferred contact details below (recommended):

- Name: Sambit Chakraborty
- Email: sambitmaths123@gmail.com
- GitHub: [sambitcodes](https://github.com/sambitcodes)
- LinkedIn: [Sambit Chakraborty](https://www.linkedin.com/in/samchak/)

---

## License

MIT License

