from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv, find_dotenv

# load model
_ = load_dotenv(find_dotenv()) 
path = os.getenv('RERANK_MODEL_PATH')
rerank_model = CrossEncoder(path)

def rerank(vec_db, user_input, top_n = 5, recall_n = 50):
    search_results = vec_db.search(user_input, recall_n)
    scores = rerank_model.predict([(user_input, doc) for doc in search_results])
    sorted_list = sorted(zip(scores,search_results['documents'][0]), key=lambda x: x[0], reverse=True)
    return [item[1] for item in sorted_list[:top_n]]