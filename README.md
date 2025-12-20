# RAG Project (FAISS + Ollama)

A local Retrieval-Augmented Generation (RAG) backend that allows users to upload documents, extract text using OCR, context text into embeddings, store them in a FAISS vector databases and then ask natural language questions. Responses are generated using Ollama (llama3 model) running locally.

This project demonstrates a minimal RAG pipeline using:
1.FastAPI(API backend)
2.Tesseract OCR
3.Sentence Transformers
4.FAISS vector search
5.Ollama LLM generation

Features:
1.Supports multiple file types for ingestion:
  - Images (PNG,JPG,JPEG) via OCR
  - PDF files using pdfplumber
  - DOCX text extraction using python-docx

2.Automatic text extraction from uploaded files.

3.File organization into separate folders:
  - uploads/ for raw uploaded files
  - processed_text/ for extracted text
  - vector_store/ for embeddings/vector index

4.Text Chunking using sliding window segmentation to retain context.

5.Embedding generation using SentenceTransformer(all-MiniLM-L6-v2).

6.Vector database implemented using FAISS for similarity search.

7.RAG-style question answering:
  - Searches FAISS for top-k similar chunks
  - Builds context from retrieved text
  - Constructs a controlled system prompt
  - Sends prompt to Ollama (LLaMA3 model) for answer generation

8.LLM inference is performed locally via Ollama using the Llama3 model for privacy-preserving document QA.

9.Returns “Not found in document” for missing information

10.FastAPI backend with two API routes:

POST /upload — upload and process documents

POST /ask — ask questions from uploaded content

## MongoDB Setup (for metadata + query logging)

MongoDB is used to store metadata associated with uploaded documents, extracted text references, timestamps, and optionally store incoming questions and generated responses.

1. Install MongoDB

Windows/macOS/Linux download:

https://www.mongodb.com/try/download/community

Verify installation:

mongod --version

2. Start MongoDB server

Windows/macOS via services
or Linux:

sudo systemctl start mongod

3. Configure MongoDB URI in project

In main.py:

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["rag_db"]
collection = db["documents"]

## Requirements

- Python 3.9+
- Ollama
- Tesseract OCR
- MongoDB installed and running

## 1.Create and activate virtual environment

In Terminal,
python -m venv venv

Activate environment,
source venv/bin/activate      # Linux / Mac

venv\Scripts\activate         # Windows

## 2. Install Python Dependencies

In Terminal,
pip install -r requirements.txt

# Tesseract OCR Setup

- Windows
Download and install from:
https://github.com/tesseract-ocr/tesseract

Update path in main.py:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

- Linux
sudo apt install tesseract-ocr

- macOS
brew install tesseract

# Ollama + Llama3 Setup

Install Ollama:
https://ollama.com/

Then run:

ollama install llama3
ollama pull llama3
ollama run llama3


Verify installed models:

ollama list

# Run FastAPI Application

Inside the project folder run:

uvicorn main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs
