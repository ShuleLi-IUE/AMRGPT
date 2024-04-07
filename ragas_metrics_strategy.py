import pandas as pd
df = pd.read_csv("./list/evaluation_questions.csv")

from vectordb_utils_shule import ShuleVectorDB
from prompt_utils import build_prompt
import pickle
import time
from openai_utils import get_completion_openai, init_openai
from logger import log_info, log_warning
from datasets import Dataset
from embedding_utils import get_embedding_bge
from sentence_transformers import CrossEncoder
import numpy as np
import math
import os

rerank_model = CrossEncoder(os.getenv('RERANK_MODEL_PATH'))

def search_db(user_input, source_type, search_strategy):
    search_results = []
    
    if search_strategy == "hnsw":
        t0 = time.perf_counter()
        search_labels = vec_db_shule.search_bge(user_input, top_n)
        texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
        t1 = time.perf_counter()
        search_results = texts
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        t2 = time.perf_counter()
        t_hnsw = t1 - t0
        t_chat = t2 - t1
        return prompt, search_results, t_hnsw, 0, t_chat, 0
    elif search_strategy == "rerank":
        
        t_hnsw, t_rerank, scores, texts, pages, titles, years, countries, ORGs = rerank(user_input, top_n, recall_n)
        search_results = texts
    
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        t0 = time.perf_counter()
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        t_chat = time.perf_counter() - t0
        return prompt, search_results, t_hnsw, t_rerank, t_chat, 0
        
    elif search_strategy == "bf":
        t0 = time.perf_counter()
        texts, pages, titles, years, countries, ORGs = bruteforce(user_input, top_n, distance_type)
        t1 = time.perf_counter()
        search_results = texts
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        t2 = time.perf_counter()
        t_bf = t1 - t0
        t_chat = t2 - t1
        return prompt, search_results, 0, 0, t_chat, t_bf
    # log_info(f"prompt content built:\n{prompt}")
    # return prompt, search_results


def rerank(user_input, top_n, recall_n):
    t0 = time.perf_counter()
    search_labels = vec_db_shule.search_bge(user_input, recall_n)
    texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
    t1 = time.perf_counter()
    # log_info(f"vec_db_shule.get_context_by_labels costs: {t1 - t0}")

    documents = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(len(pages))]
    res = rerank_model.rank(documents = documents,
                            query=user_input,
                            batch_size = 1,
                            return_documents = False,
                            show_progress_bar = False)
    t2 = time.perf_counter()
    # log_info(f"rerank_model.predict costs: {t2 - t1}")
    
    ids = [i['corpus_id'] for i in res][:top_n]
    scores = [i['score'] for i in res][:top_n]

    log_info(f"finish rerank {recall_n} texts, return highest {top_n} texts")
    return t1-t0, t2-t1, scores, [texts[i] for i in ids], [pages[i] for i in ids], [titles[i] for i in ids], [years[i] for i in ids], [countries[i] for i in ids], [ORGs[i] for i in ids]

def bruteforce(user_input, top_n, distance_type):
    embeddings_db = vec_db_shule.get_embeddings_by_labels([i for i in range(vec_db_shule.get_cnt())])
    embedding = get_embedding_bge(user_input, 8)
    
    scores = None
    if distance_type == "ip":
        scores = [np.dot(embedding, e) for e in embeddings_db]
    elif distance_type == "cosine":
        scores = [np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e)) for e in embeddings_db]
    elif distance_type == "l2":
        scores = [math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, e))) for e in embeddings_db]
    pairs = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    top_labels = [item[0] for item in pairs][:top_n]
    texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(top_labels)
    return texts, pages, titles, years, countries, ORGs
    
    
if __name__ == "__main__":
    distance_type = "ip"
    chunk = 256
    overlap = 0.4
    top_n = 8
    recall_n = 80
    # search_strategy = "hnsw"
    source = "Hybrid"
    model= "GPT-3.5"   

    print(f"chunk{chunk}, overlap{overlap} dis{distance_type} begins:")
    index_path = f"./corpus/dis_index/index_chunk{chunk}_overlap{overlap}_{distance_type}.pickle"
    with open(index_path, 'rb') as file:
            vec_db_shule = pickle.load(file)
        
    distance_type = "l2"
    
    for search_strategy in ["rerank", "bf"]:
        questions = df["Questions"].to_list()
        ground_truths = df["Groud_answer"].to_list()
        answers = []
        contexts = []
        retrieve_time = []
        completion_time = []
        t_hnsws = []
        t_reranks = []
        t_chats = []
        t_bfs = []
        # questions = questions[:2]
        # ground_truths = ground_truths[:2]

        init_openai()
        for i, query in enumerate(questions):
            print(i, "begins")
            t0 = time.time()
            prompt, search_results, t_hnsw, t_rerank, t_chat, t_bf = search_db(query, source, search_strategy)
            t1 = time.time()
            response = get_completion_openai(prompt, model=model)
            t2 = time.time()
            contexts.append(search_results) 
            answers.append(response)
            retrieve_time.append(t1 - t0)
            completion_time.append(t2 - t1)
            t_hnsws.append(t_hnsw)
            t_reranks.append(t_rerank)
            t_chats.append(t_chat)
            t_bfs.append(t_bf)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "retrieve_time": retrieve_time,
            "completion_time": completion_time, 
            "t_hnsw": t_hnsws,
            "t_rerank": t_reranks,
            "t_chat": t_chats,
            "t_bf": t_bfs
        }
        dataset = Dataset.from_dict(data)

        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            context_relevancy,
            context_entity_recall,
            answer_similarity
        )

        result = evaluate(
            dataset = dataset,
            metrics=[context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                    context_relevancy,
                    context_entity_recall,
                    answer_similarity],
            raise_exceptions=False
        )
        print("RAGAS: ", result)
        df_result = result.to_pandas()
        print(df_result)
        df_result.to_csv(f"./metrics/strategy/evaluation_chunk{chunk}_overlap{overlap}_{distance_type}_{search_strategy}.csv", na_rep='NaN')