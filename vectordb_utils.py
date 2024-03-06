import chromadb
from chromadb.config import Settings
from openai_utils import get_embedding


class InMemoryVecDB:
    
    def __init__(self, name="demo"):
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.chroma_client.reset()
        self.name = name
        self.collection = self.chroma_client.get_or_create_collection(name=name)

    def add_documents(self, documents):
        self.collection.add(
            embeddings=[get_embedding(doc) for doc in documents],
            documents=documents,
            metadatas=[{"source": self.name} for _ in documents],
            ids=[f"id_{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        """检索向量数据库"""
        print("InMemoryVecDB:search begin")
        results = self.collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=top_n
        )
        print("InMemoryVecDB:search end")
        return results['documents'][0]