# RAG Workspace

This repository houses experiments for retrieval-augmented generation (RAG) workflows. The current focus is the **CarInsuranceResearch** Streamlit application for researching and summarizing U.S. car insurance insights.

## Projects
- `CarInsuranceResearch/` � Streamlit app that ingests up to three insurance-related URLs, filters noisy content, indexes chunks into Qdrant with hybrid (dense + sparse) embeddings, and answers analytical questions via a LangChain map-reduce pipeline.

## Quick Start
1. Install Python 3.10+ and Node-free Streamlit prerequisites.
2. Clone this repo and `cd CarInsuranceResearch`.
3. Create a virtual environment: `python -m venv .venv` then activate it.
4. Install dependencies: `pip install -r requirements.txt`.
5. Copy `.env` (or create one) with `OPENAI_API_KEY` plus any optional `QDRANT_API_KEY`.
6. Launch the tool: `streamlit run app.py`.

## Configuration
- **OpenAI** � Required for both embeddings (`text-embedding-3-*`) and GPT-4o-mini chat completions.
- **Qdrant** � Provide `QDRANT_URL` and, if applicable, `QDRANT_API_KEY` to persist your vector store. Local Docker installs work out-of-the-box at `http://127.0.0.1:6333`.
- **URL sources** � The sidebar collects up to three unique research links; avoid paywalled or bot-protected pages for best results.

## Development Notes
- Document ingestion caches results using `st.cache_data`, so use the **Clear** button when changing URLs drastically.
- The retrieval chain uses MMR and FastEmbed sparse signals; tune `k`, `fetch_k`, or the chunking strategy in `app.py` to experiment.
- Contributions should include updates to project READMEs and accompanying configuration details.

## License
Distributed under the MIT License. See `LICENSE` for details.