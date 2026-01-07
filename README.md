# Research Paper QA Bot 📚

Advanced RAG system for conversational querying of research papers.

## Features

✅ **Document Processing**
- Handle multi-column PDFs, tables, figures
- Smart chunking with metadata preservation
- Automatic section & page tracking

✅ **Retrieval Augmented Generation (RAG)**
- Vector similarity search via ChromaDB
- LlamaIndex orchestration
- Explicit source citations

✅ **Multi-Model Support**
- GROQ API: Llama 3.3, Mixtral, Gemma
- Extractive: SciBERT (zero hallucination)
- Model switching in UI

✅ **Production-Grade UI**
- Custom CSS styling
- Streaming responses
- Citation panel
- Model configuration

## Quick Start

### 1. Clone & Install

\`\`\`bash
git clone <repo>
cd research-paper-qa-bot
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
\`\`\`

### 2. Configure

Copy `.env.example` to `.env` and add your API keys:

\`\`\`bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY, etc.
\`\`\`

### 3. Run

\`\`\`bash
streamlit run streamlit_app/app.py
\`\`\`

Navigate to `http://localhost:8501`

## Architecture

PDF Upload
↓
PDF Processing (Unstructured)
↓
Smart Chunking (with metadata)
↓
Embedding (sentence-transformers)
↓
Vector Store (ChromaDB)
↓
Retrieval (similarity search)
↓
Generation (GROQ LLM + context)
↓
Citation Formatting
↓
Streamlit UI


## Configuration

See `src/config.py`:
- `CHUNK_SIZE`: Document chunk size (default: 512)
- `TOP_K_RETRIEVAL`: Docs to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Min retrieval score (default: 0.5)
- `GROQ_MODELS`: Available models

## Notebooks

Run Jupyter notebooks for testing:

\`\`\`bash
jupyter notebook notebooks/
\`\`\`

- `01_PDF_Ingestion_Test.ipynb` - PDF processing
- `02_Embedding_Quality_Test.ipynb` - Embedding quality
- `03_Retrieval_Quality_Test.ipynb` - Retrieval quality
- `04_Model_Comparison.ipynb` - Model comparison
- `05_Prompt_Engineering.ipynb` - Prompt testing

## Project Structure

├── src/
│ ├── pdf_processing/ # PDF extraction
│ ├── indexing/ # Embedding & vector store
│ ├── retrieval/ # RAG retriever
│ ├── llm/ # Model integrations
│ ├── qa_pipeline/ # End-to-end pipeline
│ └── config.py # Configuration
├── streamlit_app/ # UI components
├── notebooks/ # Testing & experimentation
├── tests/ # Unit tests
└── README.md


## API Integration

### GROQ

Get free API key: https://console.groq.com

Supported models:
- `llama-3.3-70b-versatile` - Best quality
- `mixtral-8x7b-32768` - Fastest
- `gemma-7b-it` - Lightweight

### HuggingFace

Optional for embeddings and SciBERT. Get token: https://huggingface.co/settings/tokens

## Future Enhancements

- [ ] Re-ranking module
- [ ] Hallucination detection (guardrails)
- [ ] Multi-paper cross-referencing
- [ ] Feedback loop for ranking
- [ ] Web UI deployment
- [ ] API server (FastAPI)

## License

MIT
