from dotenv import load_dotenv
import  os
from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key =os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature = 0.9)

loader = CSVLoader(file_path="QA.csv", source_column="prompt",  csv_args={"delimiter": ",", "quotechar": '"'})
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_database =  FAISS.from_documents(chunks,embeddings)
# retriever = vector_database.as_retriever(search_kwargs={"k": 3})
# rdoc = retriever.get_relevant_documents("What is the refund policy?")
# rdoc
# results = retriever.invoke("How about job placement?")
# for i, d in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print(d.page_content[:300])
#     print(d.metadata)

retriever = vector_database.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 5) Prompt + LLM + parser
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an EdTech Q&A assistant. Answer ONLY using the provided context. "
     "If the answer is not in the context, say: 'I don't know based on the CSV.'"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = "Is there a job assistance?"
answer = rag_chain.invoke(question)
print(question)
print(answer)