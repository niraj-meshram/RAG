# CarInsuranceResearch

A Streamlit-based research assistant that ingests car insurance articles, cleans and chunks the content, and builds a hybrid Qdrant index so you can ask targeted questions backed by cited sources.

## Features
- **Flexible ingestion** – Load up to three arbitrary URLs (news, rate studies, regulatory pages) and normalize their content.
- **Quality filtering** – Drops boilerplate/legal noise and keeps only meaningful snippets before chunking.
- **Hybrid retrieval** – Combines OpenAI dense embeddings with FastEmbed sparse vectors inside Qdrant for better recall.
- **Map-reduce answering** – Uses LangChain LCEL primitives to summarize chunk findings and cite their sources.
- **Interactive UI** – One-page Streamlit UI with sidebar controls, cache clearing, and debugging expanders.

## Requirements
- Python 3.10+
- Streamlit, LangChain, Qdrant client, and other packages from `requirements.txt`
- OpenAI API key (for embeddings + GPT-4o-mini chat)
- Optional Qdrant Cloud/Docker endpoint for persistent storage

## Setup
1. `python -m venv .venv`
2. Activate the environment (`.venv\\Scripts\\activate` on Windows).
3. `pip install -r requirements.txt`
4. Create `.env` with:
   ```env
   OPENAI_API_KEY=sk-...
   QDRANT_URL=http://127.0.0.1:6333
   QDRANT_API_KEY= # optional when using local/no-auth setups
   ```
5. (Optional) Run `docker run -p 6333:6333 qdrant/qdrant` if you need a local Qdrant instance.

## Usage
```bash
streamlit run app.py
```
- Enter up to three URLs in the sidebar and configure your Qdrant settings.
- Click **Run Research** to scrape, clean, chunk, and index the data.
- Use the QA section to ask domain questions and receive cited answers.
- Use **Clear** whenever you change datasets significantly to reset cached state.

## Customization
- Adjust chunk size/overlap in `postprocess_docs` to better fit your content.
- Tune retriever parameters in `get_optimized_retriever` (`k`, `fetch_k`, `lambda_mult`).
- Swap embedding models via the sidebar select box or the defaults in `app.py`.

## Troubleshooting
- **Connection errors** – Verify the Qdrant URL/API key combo using the built-in healthcheck message.
- **Rate limits** – Lower the number of URLs or chunks when hitting OpenAI limits; each chunk triggers embeds + completion calls.
- **Stale cache** – Hit **Clear** to invalidate `st.cache_data` entries when iterating on preprocessing logic.

## License
This project inherits the MIT License from the workspace root.
