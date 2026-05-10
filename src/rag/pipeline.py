"""
rag/pipeline.py — Modular RAG pipeline.
Supports swappable: embedding models, vector stores, chunking strategies.
"""
import os, re
from typing import List, Tuple, Optional


class RAGPipeline:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 256, chunk_overlap: int = 32,
                 db_path: str = "data/chromadb"):
        from sentence_transformers import SentenceTransformer
        import chromadb
        self.embedder     = SentenceTransformer(embed_model)
        self.client       = chromadb.PersistentClient(path=db_path)
        self.col          = self.client.get_or_create_collection("docs")
        self.chunk_size   = chunk_size
        self.chunk_overlap= chunk_overlap
        self._id_counter  = 0

    def _chunk(self, text: str) -> List[str]:
        words  = text.split()
        chunks = []
        step   = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i: i + self.chunk_size]))
        return [c for c in chunks if len(c.strip()) > 20]

    def ingest_text(self, text: str) -> int:
        text   = re.sub(r'\s+', ' ', text).strip()
        chunks = self._chunk(text)
        if not chunks:
            return 0
        embs   = self.embedder.encode(chunks, show_progress_bar=False)
        ids    = [f"doc_{self._id_counter + i}" for i in range(len(chunks))]
        self._id_counter += len(chunks)
        self.col.add(documents=chunks, embeddings=embs.tolist(), ids=ids)
        return len(chunks)

    def ingest_pdf(self, path: str) -> int:
        try:
            import fitz
            doc  = fitz.open(path)
            text = " ".join(p.get_text() for p in doc)
            doc.close()
            return self.ingest_text(text)
        except ImportError:
            raise ImportError("pip install PyMuPDF")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        emb     = self.embedder.encode([query])
        results = self.col.query(query_embeddings=emb.tolist(), n_results=k)
        return results["documents"][0] if results["documents"] else []
