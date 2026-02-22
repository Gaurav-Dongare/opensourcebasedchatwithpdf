'''
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

#vectorembedding model call
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
)

#vectordb connection , here we use from exisitng collection that we created in indexing process
vector_db = QdrantVectorStore.from_existing_collection(embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_RAG",
    force_recreate = True)

user_query = input("What would you like to know?")

#similarity search: 
search_results = vector_db.similarity_search(query = user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage number: {result.metadata['page_label']}\n File Location: {result.metadata['soure']}" for result in search_results])

SYSTEM_PROMPT = f"""
Your are a helpful AI assitant, who answeres userquery based on the available context retrieved from a pdf file along with page_content and page_number. 

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

context:
{context}
"""
'''


from fastapi import FastAPI, Body
from ollama import Client
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore


app = FastAPI()

# 1. Setup Connections
# Connect to Ollama (Running in Docker)
ollama_client = Client(host="http://localhost:11434")

# Setup Embedding model for Qdrant search
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Connect to Qdrant (Running in Docker)
# Note: Set force_recreate=False here so it doesn't delete your data on every API call!
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_RAG"
)

@app.post("/chat")
def chat(message: str = Body(..., embed=True)):
    # 2. Retrieval: Search Qdrant for relevant PDF chunks
    search_results = vector_db.similarity_search(query=message, k=3)

    # 3. Context Preparation
    # Be careful with metadata keys; PyPDFLoader usually uses 'page' and 'source'
    context_list = []
    for result in search_results:
        content = result.page_content
        page = result.metadata.get('page', 'N/A')
        source = result.metadata.get('source', 'Unknown')
        context_list.append(f"Content: {content}\nPage: {page}\nSource: {source}")
    
    context = "\n\n---\n\n".join(context_list)

    # 4. Prompt Engineering
    system_prompt = f"""
    You are a helpful AI assistant. Answer the user's question ONLY using the context below.
    Always mention the page number where you found the information.

    Context:
    {context}
    """

    # 5. API Call to Ollama (Gemma)
    response = ollama_client.chat(
        model="gemma:2b", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )

    return {
        "answer": response.message.content,
        "sources": [res.metadata for res in search_results]
    }