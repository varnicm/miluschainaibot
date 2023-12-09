import config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

loader = WebBaseLoader([
    "https://www.atu.edu/about/index.php",
])

docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
docs = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()

vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT}
)


