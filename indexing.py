# Document Loader : 
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from ollama import Client

#pdf_path = Path(__file__).parent / "nodejs.pdf"



#1. load this file in python program https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader

file_path = "Rag/jsnode.pdf"
loader = PyPDFLoader(file_path)
docs =loader.load()

#2. Splitting docs into smaller chunks CHUNKING https://docs.langchain.com/oss/python/integrations/splitters


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents=docs)

#3. Creating vector embeddings https://docs.langchain.com/oss/python/integrations/text_embedding/ollama

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434" # a very important step. this points directly to ollma model in the docker container, where nomic embed text is pulled from powershell temrinal
)

#4. using qdrant db to store vector embeddings for retrieval https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_RAG",
    force_recreate = True
) 

print("Indexing of Docs done using local Ollama embeddings!")
