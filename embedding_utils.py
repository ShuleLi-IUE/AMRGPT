from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv, find_dotenv

# load model
_ = load_dotenv(find_dotenv()) 
path = os.getenv('EMBED_MODEL_PATH')
embedding_model = SentenceTransformer(path)

# bge-large-zh-v1.5
# dim: 1024
def get_embedding_bge(text):
    data = embedding_model.encode(text, normalize_embeddings=True).tolist()
    return data
