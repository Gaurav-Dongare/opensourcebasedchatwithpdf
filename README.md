# Open Source Based Chat with PDF

This is a local RAG (Retrieval-Augmented Generation) system built with **FastAPI**, **Ollama**, and **Qdrant**. It allows you to index PDF documents into a vector database and query them using open-source LLMs while keeping all data private on your local machine.

## 🧠 Theory & Architecture

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is an AI framework that improves the quality of Large Language Model (LLM) responses by grounding the model on external sources of knowledge. Instead of relying solely on the LLM's static training data, RAG searches a database of your private documents, retrieves the most relevant facts, and feeds them to the LLM to generate an accurate, hallucination-free answer.

### How This System Works (Under the Hood)
This project is broken down into two main phases matching the provided scripts:

**1. The Indexing Phase (`indexing.py`)**
* **Document Loading:** Uses LangChain's `PyPDFLoader` to extract raw text from your PDF.
* **Chunking:** Uses `RecursiveCharacterTextSplitter` to break the document into manageable 1000-character chunks with a 200-character overlap. This ensures that concepts spanning across paragraphs aren't cut off.
* **Vector Embeddings:** Converts the text chunks into mathematical vectors using Ollama's `nomic-embed-text` model. These vectors capture the semantic meaning of the text.
* **Storage:** Saves these embeddings into **Qdrant**, a high-performance vector database running locally via Docker.

**2. The Retrieval & Generation Phase (`chat_with_pdf.py`)**
* **User Query:** You ask a question via the FastAPI endpoint.
* **Semantic Search:** The system converts your question into a vector and performs a similarity search in Qdrant to retrieve the top 3 most relevant text chunks from the PDF.
* **Prompt Engineering:** The retrieved text, along with its metadata (page numbers and source), is injected into a strict system prompt.
* **LLM Generation:** The augmented prompt is sent to the `gemma:2b` model via Ollama. The model generates a conversational response based *only* on the provided context, telling you exactly which page to look at.

---

## Installation

### 1. Prerequisites
Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [Python 3.10+](https://www.python.org/downloads/) installed.

### 2. Setup Infrastructure
Start the **Qdrant** vector database service using Docker Compose:

```bash
docker-compose up -d
```
### 3. Setup Ollama
Pull the gemma:2b model (or any model of your choice based on your system specs) inside the Ollama container:
```bash
docker exec -it ollama ollama pull gemma:2b
```
### 3. Install Pyhton Dependencies
```bash
pip install -r requirements.txt
```


## How to run?
### 1. Indexing the pdf
Place your PDF file (e.g., jsnode.pdf) in the project folder and run the indexing script to create vector embeddings in Qdrant:
```bash
python indexing.py
```
### 2. Starting the API

```bash
uvicorn chat_with_pdf:app --reload
```

### 3. Querying the PDF
ou can interact with the system via the Swagger UI at http://127.0.0.1:8000/docs

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT]

