from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import os
from dotenv import load_dotenv, find_dotenv

# load model
_ = load_dotenv(find_dotenv()) 
path = os.getenv('MULTILANG_EMBED_MODEL_PATH')

# embedding_model = SentenceTransformer(path)
embedding_model = BGEM3FlagModel(path,
                      use_fp16=False)

# bge-large-en-v1.5
# dim: 1024
# mono language
# def get_embedding_bge_15(text):
#     data = embedding_model.encode(text, normalize_embeddings=True).tolist()
#     return data

# bge-m3
# dim: 1024
# multi language
def get_embedding_bge_m3(text):
    data = embedding_model.encode(text, 
                                  return_dense = True,
                                  return_sparse= False,
                                  return_colbert_vecs = False)['dense_vecs'].flatten().tolist()
    return data

def get_embedding_bge(text):
    return get_embedding_bge_m3(text)