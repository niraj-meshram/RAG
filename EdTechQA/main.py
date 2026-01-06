from dotenv import load_dotenv
import  os
from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
api_key =os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature = 0.9)

loader = CSVLoader(file_path="QA.csv", source_column="prompt",  csv_args={"delimiter": ",", "quotechar": '"'})
docs = loader.load()
# print(len(docs))
# print(docs[1].page_content)
# print(docs[1].metadata)
#
# print(docs[0].page_content[:500])
# print(docs[0].metadata.keys())
# print(docs)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = embeddings.embed_query("hello")
print(len(vec))