from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import re
import pymupdf as fitz
import faiss
import nltk
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

nltk.download('punkt', quiet=True)

PDF_FILE = "HSC26_Bangla_1st_paper.pdf"
CHUNK_SIZE = 350

# Text Extraction
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
    return full_text

# Data Cleaning & Chunking
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[|_]+', '', text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 20]
    return paras

def chunk_long_paragraphs(paragraphs: List[str], chunk_size=CHUNK_SIZE) -> List[str]:
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            sentences = nltk.sent_tokenize(para, language="english")
            if not sentences or len(sentences) == 1:
                sentences = re.split(r"(?<=[।?])\s+", para)
            temp = ""
            for sent in sentences:
                if len(temp) + len(sent) < chunk_size:
                    temp += sent + " "
                else:
                    if temp.strip():
                        chunks.append(temp.strip())
                    temp = sent + " "
            if temp.strip():
                chunks.append(temp.strip())
    return [c for c in chunks if c]

def embed_texts(model, texts: List[str], batch_size=32) -> np.ndarray:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(embs)
    return np.vstack(embeddings)

class VectorDB:
    def __init__(self, embeddings, texts):
        self.embed_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.texts = texts
        self.index.add(embeddings)

    def search(self, query_emb, top_k=1):
        D, I = self.index.search(query_emb.astype('float32'), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            result_text = self.texts[idx]
            results.append((result_text, float(score)))
        return results

def build_knowledge_base(pdf_path: str):
    pdf_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(pdf_text)
    paragraphs = split_paragraphs(clean)
    chunks = chunk_long_paragraphs(paragraphs)
    return chunks

# ----------------- FastAPI Setup --------------------

class QueryRequest(BaseModel):
    question: str

app = FastAPI(title="Multilingual RAG QA API")

# One-time init
print("Loading knowledge base and embeddings...")
kb_chunks = build_knowledge_base(PDF_FILE)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
kb_embeddings = embed_texts(model, kb_chunks)
vector_db = VectorDB(kb_embeddings, kb_chunks)

@app.post("/ask")
async def ask_question(req: QueryRequest):
    query = req.question.strip()
    q_emb = model.encode([query], normalize_embeddings=True)
    best_chunks = vector_db.search(q_emb, top_k=1)
    answer_text = best_chunks[0][0].strip()
    return {"question": query, "answer": answer_text}

@app.get("/")
async def read_root():
    return {"message": "Multilingual RAG API. POST to /ask with {'question': <your question>}"}

if __name__ == "__main__":
    uvicorn.run("api_rag:app", host="0.0.0.0", port=8000, reload=True)
