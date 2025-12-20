from fastapi import FastAPI, UploadFile, Body
import uuid, os
from PIL import Image
import pytesseract
import pdfplumber
import docx
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone
from googlesearch import search
from bs4 import BeautifulSoup
from readability import Document

#MONGODB ATLAS SETUP
MONGO_URI = "mongodb+srv://tankardhanashree05_db_user:qtlBmgKu7m9bppWd@cluster0.zdc8mpn.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["document_db"]
collection = db["documents"]

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_DIR = "uploads"
TEXT_DIR = "processed_text"
VECTOR_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

app = FastAPI()

#EMBEDDINGS 
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

documents = []   # stores chunk text
doc_ids = []     # stores document IDs

#UTILS 
def extract_text(file_path: str, filename: str) -> str:
    if filename.endswith((".png", ".jpg", ".jpeg")):
        return pytesseract.image_to_string(Image.open(file_path))

    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# GOOGLE SEARCH FALLBACK
def google_fallback(query, num_results=5):
    urls = list(search(query + " wikipedia", num_results=num_results))
    contents = []
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            doc = Document(r.text)
            text = doc.get_clean_html()
            soup = BeautifulSoup(text, "html.parser")
            main_text = soup.get_text(separator="\n")
            contents.append(main_text[:2000])
        except:
            continue
    return "\n\n".join(contents)

#FILE UPLOAD 
@app.post("/upload")
async def upload_file(file: UploadFile):
    doc_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{doc_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(file_path, file.filename.lower())
    
    if not text.strip():
        return {"error": "No text extracted from file"}

    text_file = f"{TEXT_DIR}/{doc_id}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)

    index.add(np.array(embeddings).astype("float32"))
    documents.extend(chunks)
    doc_ids.extend([doc_id] * len(chunks))

    # save metadata to MongoDB
    collection.insert_one({
    "doc_id": doc_id,
    "filename": file.filename,
    "file_path": file_path,
    "text_file": text_file,
    "chunks_created": len(chunks),
    "uploaded_at": datetime.now(timezone.utc)
    })

    return {
        "message": "File processed successfully",
        "doc_id": doc_id,
        "chunks_created": len(chunks)
    }

#QUESTION ANSWERING
@app.post("/ask")
async def ask_question(question: str):

    q_embedding = embedder.encode([question]).astype("float32")
    D, I = index.search(q_embedding, k=3)

    threshold = 10.0   

    top_distance = float(D[0][0])

    # case 1: document match exists
    if top_distance < threshold:
        context = "\n\n".join([documents[i] for i in I[0]])
        source = "documents"

        prompt = f"""
SYSTEM:
Use the context to answer the question.
If answer not found, respond: "Not found in document."

CONTEXT:
{context}

QUESTION:
{question}
"""
    
    # case 2: google fallback
    else:
        context = google_fallback(question)
        source = "google"

        if not context.strip():
            context = "No usable webpage text extracted."
        
        prompt = f"""
SYSTEM:
You are a web-augmented assistant.
Answer based only on webpage context.

WEB CONTEXT:
{context}

QUESTION:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json().get("response","")

    return {
        "answer": answer,
        "source": source,
        "distance": top_distance
    }
