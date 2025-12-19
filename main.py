from fastapi import FastAPI, UploadFile
import uuid, os
from PIL import Image
import pytesseract
import pdfplumber
import docx
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

    context = "\n\n".join([documents[i] for i in I[0]])

    prompt = f"""
SYSTEM:
You are a document-based assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "Not found in document."

CONTEXT:
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

    return {
        "answer": response.json().get("response", "")
    }
