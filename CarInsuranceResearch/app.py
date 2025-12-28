import streamlit as st
from urllib.parse import urlparse
from collections import Counter, defaultdict

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------
# Helpers
# ---------------------------
def is_valid_url(u: str) -> bool:
    if not u:
        return False
    p = urlparse(u)
    return p.scheme in ("http", "https") and bool(p.netloc)


@st.cache_data(show_spinner=False)
def load_docs(urls: tuple[str, ...]) -> list[Document]:
    """Load each URL separately and stamp metadata['source']=url on every element."""
    all_docs: list[Document] = []
    for url in urls:
        loader = UnstructuredURLLoader(urls=[url], mode="elements")
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = url
        all_docs.extend(docs)
    return all_docs


def postprocess_docs(elements: list[Document]) -> list[Document]:
    # 1) filter tiny / low-signal bits
    elements = [d for d in elements if len(d.page_content.strip()) >= 120]

    # 2) drop boilerplate patterns (tune as needed)
    bad_snippets = (
        "cookie",
        "advertis",
        "newsletter",
        "privacy policy",
        "terms of use",
        "terms and conditions",
        "do not sell",
        "copyright",
    )
    elements = [
        d for d in elements
        if not any(b in d.page_content.lower() for b in bad_snippets)
    ]

    # 3) merge elements per source URL into 1 doc each
    by_source: dict[str, list[str]] = defaultdict(list)
    for d in elements:
        src = (d.metadata.get("source") or "unknown").strip() or "unknown"
        by_source[src].append(d.page_content.strip())

    merged_docs: list[Document] = []
    for src, parts in by_source.items():
        merged_text = "\n\n".join(parts)
        merged_docs.append(Document(page_content=merged_text, metadata={"source": src}))

    # 4) chunk the merged docs for RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(merged_docs)


def qdrant_healthcheck(url: str, api_key: str | None) -> tuple[bool, str]:
    """Friendly connectivity check."""
    try:
        c = QdrantClient(url=url, api_key=(api_key or None))
        c.get_collections()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def build_qdrant_index(
    docs: list[Document],
    *,
    qdrant_url: str,
    qdrant_api_key: str | None,
    collection_name: str,
    recreate: bool,
    embedding_model: str,
) -> QdrantVectorStore:
    """
    Creates/updates a Qdrant collection from LangChain Documents.
    Uses hybrid retrieval (dense + sparse).
    """
    dense = OpenAIEmbeddings(model=embedding_model)
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    client = QdrantClient(url=qdrant_url, api_key=(qdrant_api_key or None))

    if recreate:
        try:
            client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

    vector_store = QdrantVectorStore.from_documents(
        docs,
        embedding=dense,
        sparse_embedding=sparse,
        url=qdrant_url,
        api_key=(qdrant_api_key or None),
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
        prefer_grpc=False,
    )
    return vector_store


def get_optimized_retriever(vector_store: QdrantVectorStore, k: int = 6):
    # MMR for diversity; fetch_k controls how many candidates are pulled before reranking
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(20, k * 4),
            "lambda_mult": 0.5,
        },
    )


def answer_map_reduce_pure_latest(vector_store: QdrantVectorStore, question: str, k: int = 6):
    """
    Pure latest LangChain (LCEL-style): Retrieval -> Map (batch) -> Reduce (combine).
    No langchain.chains / no langchain-classic.
    """
    retriever = get_optimized_retriever(vector_store, k=k)
    docs = retriever.invoke(question)

    # Include SOURCE in text so the LLM can cite it
    contexts = [
        f"SOURCE: {(d.metadata.get('source') or 'unknown')}\n\n{d.page_content}"
        for d in docs
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    map_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an insurance research assistant.\n"
            "Given ONE excerpt, extract only facts relevant to the question.\n"
            "If not relevant, output: NOT_RELEVANT.\n\n"
            "Question:\n{question}\n\n"
            "Excerpt:\n{context}\n\n"
            "Return bullet facts with the SOURCE url for each fact."
        ),
    )
    map_chain = map_prompt | llm | parser

    # Efficient map step over all retrieved chunks
    mapped = map_chain.batch([{"context": c, "question": question} for c in contexts])

    combine_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template=(
            "Combine the findings into a final answer.\n"
            "- Deduplicate\n"
            "- If sources conflict, mention it\n"
            "- Keep SOURCE urls next to claims\n\n"
            "Question:\n{question}\n\n"
            "Findings:\n{summaries}\n\n"
            "Final answer:"
        ),
    )
    reduce_chain = combine_prompt | llm | parser

    final_answer = reduce_chain.invoke({"summaries": "\n\n".join(mapped), "question": question})
    sources = sorted({(d.metadata.get("source") or "unknown") for d in docs})
    return final_answer, sources, docs


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Insurance Research Tool", layout="wide")
st.title("Insurance Research Tool")

st.sidebar.header("Input URLs")
url1 = st.sidebar.text_input("URL 1", placeholder="https://www.valuepenguin.com/best-cheap-car-insurance-california")
url2 = st.sidebar.text_input("URL 2", placeholder="https://www.moneygeek.com/insurance/auto/best-cheap-car-insurance-santa-clara-ca/")
url3 = st.sidebar.text_input("URL 3", placeholder="https://www.carinsurance.com/state-car-insurance-rates")

urls = [url1.strip(), url2.strip(), url3.strip()]
urls = [u for u in urls if u]
urls = list(dict.fromkeys(urls))  # dedupe

st.sidebar.header("Vector DB (Qdrant)")
qdrant_url = st.sidebar.text_input("Qdrant URL", value="http://127.0.0.1:6333")
qdrant_api_key = st.sidebar.text_input("Qdrant API Key (optional)", value="", type="password")
collection_name = st.sidebar.text_input("Collection name", value="insurance_research")
recreate_collection = st.sidebar.checkbox("Recreate collection on each run", value=True)

embedding_model = st.sidebar.selectbox(
    "Embedding model",
    ["text-embedding-3-small", "text-embedding-3-large"],
    index=0,
)

col_a, col_b = st.sidebar.columns(2)
submitted = col_a.button("Run Research", use_container_width=True)
clear = col_b.button("Clear", use_container_width=True)

main_placeholder = st.empty()

if clear:
    st.cache_data.clear()
    st.session_state.clear()
    st.success("Cleared cache and session.")

# Load state
docs = st.session_state.get("docs")
raw_count = st.session_state.get("raw_docs_count")
vector_store = st.session_state.get("vector_store")

# Run pipeline
if submitted:
    invalid = [u for u in urls if not is_valid_url(u)]
    if invalid or len(urls) == 0:
        st.error("Please enter at least one valid http/https URL.")
        if invalid:
            st.write("Invalid entries:", invalid)
    else:
        ok, msg = qdrant_healthcheck(qdrant_url, qdrant_api_key)
        if not ok:
            st.error("Qdrant is not reachable.")
            st.code(f"URL: {qdrant_url}\nError: {msg}")
        else:
            main_placeholder.text("Loading pages...✅✅✅")
            with st.spinner("Loading pages..."):
                raw_docs = load_docs(tuple(urls))

            main_placeholder.text("Filtering + merging + chunking...✅✅✅")
            with st.spinner("Filtering + merging + chunking..."):
                chunks = postprocess_docs(raw_docs)

            st.session_state["docs"] = chunks
            st.session_state["raw_docs_count"] = len(raw_docs)

            main_placeholder.text("Embedding + uploading to Qdrant...✅✅✅")
            with st.spinner("Embedding + uploading to Qdrant..."):
                vs = build_qdrant_index(
                    chunks,
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key,
                    collection_name=collection_name,
                    recreate=recreate_collection,
                    embedding_model=embedding_model,
                )
                st.session_state["vector_store"] = vs

            # refresh local refs
            docs = st.session_state.get("docs")
            raw_count = st.session_state.get("raw_docs_count")
            vector_store = st.session_state.get("vector_store")

# Display indexing summary
if docs:
    st.success(f"Loaded {raw_count} elements → {len(docs)} chunks (RAG-ready).")

    if vector_store:
        st.success(f"✅ Indexed into Qdrant collection: **{collection_name}**")

    lengths = [len(d.page_content) for d in docs]
    st.write(
        f"Chunk length (chars): avg **{sum(lengths)//len(lengths)}**, "
        f"min **{min(lengths)}**, max **{max(lengths)}**"
    )

    sources = [(d.metadata.get("source") or "unknown").strip() or "unknown" for d in docs]
    counts = Counter(sources)
    with st.expander("Chunks per source", expanded=False):
        for src, c in counts.most_common():
            st.write(f"- {c} • {src}")

    st.subheader("Preview (first 6 chunks)")
    for i, d in enumerate(docs[:6], start=1):
        src = d.metadata.get("source", "")
        with st.expander(f"Chunk {i} • source: {src}", expanded=(i == 1)):
            st.write(d.page_content[:2000])
else:
    st.caption("Enter URLs in the sidebar and click **Run Research**.")

# RAG QA (only show if vector_store is ready)
st.divider()
st.subheader("Ask (Optimized RAG: Hybrid + MMR + Map-Reduce)")

if not vector_store:
    st.info("Run **Run Research** first to build the Qdrant index, then ask a question.")
else:
    rag_q = st.text_input("Question", "What factors affect car insurance rates?", key="rag_q")
    k = st.slider("Top-k retrieved chunks", 3, 12, 6, key="rag_k")

    if st.button("Answer", key="rag_answer_btn"):
        with st.spinner("Retrieving (hybrid+MMR) + map-reduce answering..."):
            answer, used_sources, retrieved = answer_map_reduce_pure_latest(vector_store, rag_q, k=k)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Sources"):
            for s in used_sources:
                st.write(f"- {s}")

        with st.expander("Retrieved chunks (debug)"):
            for d in retrieved:
                st.markdown(f"**Source:** {d.metadata.get('source','')}")
                st.write(d.page_content[:500])
                st.divider()
