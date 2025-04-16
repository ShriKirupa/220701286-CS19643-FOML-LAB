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
            self.index = faiss.IndexFlatL2(384)
    


    def add_texts(self, docs, source_id):
        chunks = [{'chunk': doc, 'source': source_id} for doc in docs]
        embeddings = self.model.encode([doc['chunk'] for doc in chunks])
        self.index.add(embeddings)
        self.texts.extend(chunks)
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, "wb") as f:
            pickle.dump(self.texts, f)

    def query(self, q, k=5, allowed_sources=None):
        q_vec = self.model.encode([q])
        D, I = self.index.search(q_vec, k)
        results = []
        for i in I[0]:
            if i < len(self.texts):
                item = self.texts[i]
                if allowed_sources is None or item["source"] in allowed_sources:
                    results.append(item["chunk"])
        return results
    
def clean_context(text):
        # Remove things like [Applause], [Music], multiple spaces, etc.
         text = re.sub(r"\[.*?\]", "", text)
         text = re.sub(r"\s+", " ", text)
         return text.strip()
