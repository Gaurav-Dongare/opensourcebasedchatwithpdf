# Open Source Based Chat with PDF

This is a local RAG (Retrieval-Augmented Generation) system built with **FastAPI**, **Ollama**, and **Qdrant**. It allows you to index PDF documents into a vector database and query them using open-source LLMs while keeping all data private on your local machine.



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

