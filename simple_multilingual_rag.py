import re
import pymupdf as fitz
import faiss
import nltk
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm

nltk.download('punkt')

PDF_FILE = "HSC26_Bangla_1st_paper.pdf"  # Downloaded and placed locally
CHUNK_SIZE = 150  # characters; adjust as needed

# -----------------------------------------
# PDF Text Extraction
# -----------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
    return full_text

# -----------------------------------------
# Preprocessing & Paragraph Chunking
# -----------------------------------------
def clean_text(text: str) -> str:
    # Remove excessive whitespace and unwanted characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[|_]+', '', text)        # Table artifacts
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    # Split by double newlines, commonly used in Bengali docs (may need to adapt)
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 50]
    return paras

def chunk_long_paragraphs(paragraphs: List[str], chunk_size=CHUNK_SIZE) -> List[str]:
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # Split into chunks at sentence boundaries
            sentences = nltk.sent_tokenize(para, language="english")
            # Try Bengali sentence splitting
            if not sentences or len(sentences) == 1:
                sentences = re.split(r"(?<=[।?])\s+", para)  # Split at Bengali fullstop/dari or questionmark
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

# -----------------------------------------
# Embedding Model Initialization
# -----------------------------------------
def embed_texts(model, texts: List[str], batch_size=32) -> np.ndarray:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(embs)
    return np.vstack(embeddings)

# -----------------------------------------
# FAISS Vector DB
# -----------------------------------------
class VectorDB:
    def __init__(self, embeddings, texts):
        self.embed_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.texts = texts
        self.index.add(embeddings)

    def search(self, query_emb, top_k=3):
        D, I = self.index.search(query_emb.astype('float32'), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            result_text = self.texts[idx]
            results.append((result_text, float(score)))
        return results

# -----------------------------------------
# Main RAG pipeline
# -----------------------------------------
def build_knowledge_base(pdf_path: str):
    print("Extracting PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(pdf_text)
    paragraphs = split_paragraphs(clean)
    chunks = chunk_long_paragraphs(paragraphs)
    return chunks

def main():
    # Load and chunk KB
    if not os.path.exists(PDF_FILE):
        print(f"ERROR: PDF file '{PDF_FILE}' not found. Place it in the script directory.")
        return

    kb_chunks = build_knowledge_base(PDF_FILE)
    print(f"Knowledgebase: {len(kb_chunks)} chunks.")

    # Load multilingual embedding model
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Build vector DB
    kb_embeddings = embed_texts(model, kb_chunks)
    vector_db = VectorDB(kb_embeddings, kb_chunks)

    # Chat Loop
    print("\n==== Multilingual RAG QA ====\nType your question (English/Bangla). Ctrl+C to exit.")
    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            q_emb = model.encode([query], normalize_embeddings=True)
            best_chunks = vector_db.search(q_emb, top_k=3)
            print("\n[Top Retrieved:]")
            for i, (txt, score) in enumerate(best_chunks, 1):
                print(f"[{i}] {txt.strip()} (score={score:.3f})")
            print("\nAnswer:", best_chunks[0][0].strip())
        except KeyboardInterrupt:
            print("\nExit.")
            break

if __name__ == '__main__':
    main()
