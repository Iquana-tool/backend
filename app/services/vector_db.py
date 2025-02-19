import faiss
import numpy as np
import os
import pickle

# Define paths for storing FAISS index and image mappings
VECTOR_DB_PATH = "database/vector_db.index"
IMAGE_MAP_PATH = "database/image_map.pkl"
UPLOADS_DIR = "uploads"  # Directory where images are stored

class VectorDatabase:
    def __init__(self, embedding_dim=256*64*64):  # Supports flattened embeddings
        """Initialize the vector database with FAISS."""
        self.embedding_dim = embedding_dim

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.image_map = {}

        # Load existing database if available
        if os.path.exists(VECTOR_DB_PATH) and os.path.exists(IMAGE_MAP_PATH):
            self.load()
        else:
            print("🟡 No existing vector database found. Starting fresh.")

    def add_embedding(self, image_path, embedding):
        """Adds an image embedding to the FAISS vector database."""
        try:
            embedding = np.array(embedding).astype("float32").reshape(1, -1)  # Ensure 1D shape

            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"❌ Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[1]}.")

            # Add embedding to FAISS index
            self.index.add(embedding)

            # Store image path with index position
            self.image_map[self.index.ntotal - 1] = image_path

            # Save updated database
            self.save()

            # Debugging logs
            print(f"✅ Added embedding for: {image_path}")
            self.debug_index()
        except Exception as e:
            print(f"❌ Failed to add embedding: {e}")

    def search(self, query_embedding, top_k=5):
        """Finds similar images in the vector database and returns accessible URLs."""
        try:
            if self.index.ntotal == 0:
                raise ValueError("Vector database is empty. Upload images first.")

            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

            if query_embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"❌ Query embedding dimension mismatch. Expected {self.embedding_dim}, got {query_embedding.shape[1]}.")

            distances, indices = self.index.search(query_embedding, top_k)

            # 🔥 Convert absolute file paths to API-accessible URLs
            results = [
                f"http://127.0.0.1:8000/uploads/{os.path.basename(self.image_map[idx])}"
                if idx in self.image_map else "Unknown"
                for idx in indices[0]
            ]

            # Debugging logs
            print(f"🔍 Query Embedding shape: {query_embedding.shape}")
            print(f"📌 Top {top_k} Matches: {results}")

            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def save(self):
        """Saves the FAISS index and image mapping."""
        try:
            faiss.write_index(self.index, VECTOR_DB_PATH)
            with open(IMAGE_MAP_PATH, "wb") as f:
                pickle.dump(self.image_map, f)
            print("💾 Vector database saved successfully!")
        except Exception as e:
            print(f"❌ Failed to save vector database: {e}")

    def load(self):
        """Loads the FAISS index and image mapping from disk."""
        try:
            self.index = faiss.read_index(VECTOR_DB_PATH)
            with open(IMAGE_MAP_PATH, "rb") as f:
                self.image_map = pickle.load(f)
            print("✅ Vector database loaded successfully!")
            self.debug_index()
        except Exception as e:
            print(f"❌ Failed to load vector database: {e}")

    def debug_index(self):
        """Prints the size and contents of the vector database."""
        print(f"📊 FAISS Index Size: {self.index.ntotal}")
        print(f"📂 Stored Image Paths: {self.image_map}")

# Initialize the vector database
vector_db = VectorDatabase()
vector_db.debug_index()  # Check database state on startup
