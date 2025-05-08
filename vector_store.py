import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle
import re
from ollama_chat import call_deepseek

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_file = "vector.index"
        self.data_file = "docs.pkl"
        self.texts = []  # Each item: {'chunk': ..., 'source': ...}

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.data_file, "rb") as f:
                self.texts = pickle.load(f)
        else:
            # Use normalized inner product index for better semantic search
            self.index = faiss.IndexFlatIP(384)

    def add_texts(self, docs, source_id):
        chunks = [{'chunk': doc, 'source': source_id} for doc in docs]
        embeddings = self.model.encode([doc['chunk'] for doc in chunks], normalize_embeddings=True)
        self.index.add(embeddings)
        self.texts.extend(chunks)
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, "wb") as f:
            pickle.dump(self.texts, f)

    def query(self, query, k=5, allowed_sources=None):
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        distances, indices = self.index.search(query_embedding, k=10)  # Fetch more initially

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            if 0 <= idx < len(self.texts):
                if allowed_sources is None or self.texts[idx]['source'] in allowed_sources:
                    results.append((idx, distances[0][list(indices[0]).index(idx)]))

        # For inner product, higher score = better match
        results.sort(key=lambda x: -x[1])  # Sort descending

        final_texts = []
        for i, _ in results[:k]:
            final_texts.append(self.texts[i])
        return final_texts

def clean_context(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
