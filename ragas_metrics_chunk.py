import pandas as pd
df = pd.read_csv("./list/evaluation_questions.csv")

from vectordb_utils_shule import ShuleVectorDB
from prompt_utils import build_prompt
import pickle
import time
from openai_utils import get_completion_openai, init_openai
from logger import log_info, log_warning
from datasets import Dataset

def search_db(user_input, source_type):
    search_results = []
    
    if search_strategy == "hnsw":
        search_labels = vec_db_shule.search_bge(user_input, top_n)
        texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
        search_results = texts
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        
    elif search_strategy == "rerank":
        scores, texts, pages, titles, years, countries, ORGs = rerank(user_input, top_n, recall_n)
        search_results = texts
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        
    elif search_strategy == "fusion":
        log_warning("Not support yet.")
        return
            
    log_info(f"prompt content built:\n{prompt}")
    return prompt, search_results


def rerank(user_input, top_n, recall_n):
    search_labels = vec_db_shule.search_bge(user_input, recall_n)
    t0 = time.time()
    texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
    t1 = time.time()
    log_info(f"vec_db_shule.get_context_by_labels costs: {t1 - t0}")

    documents = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(len(pages))]
    res = rerank_model.rank(documents = documents,
                            query=user_input,
                            batch_size = 1,
                            return_documents = False,
                            show_progress_bar = False)
    t2 = time.time()
    log_info(f"rerank_model.predict costs: {t2 - t1}")
    
    ids = [i['corpus_id'] for i in res][:top_n]
    scores = [i['score'] for i in res][:top_n]

    log_info(f"finish rerank {recall_n} texts, return highest {top_n} texts")
    return scores, [texts[i] for i in ids], [pages[i] for i in ids], [titles[i] for i in ids], [years[i] for i in ids], [countries[i] for i in ids], [ORGs[i] for i in ids]


for chunk in [512]:
    for overlap in [0.2, 0.4]:
        if chunk == 512 and overlap in [0, 0.1]: continue
        print(f"chunk{chunk}, overlap{overlap} begins:")
        index_path = f"./corpus/chunks/index_chunk{chunk}_overlap{overlap}.pickle"
        top_n = 8
        recall_n = 80
        search_strategy = "hnsw"
        source = "Hybrid"
        model= "GPT-3.5"

        with open(index_path, 'rb') as file:
                vec_db_shule = pickle.load(file)
            
        
        questions = df["Questions"].to_list()
        ground_truths = df["Groud_answer"].to_list()
        answers = []
        contexts = []
        retrieve_time = []
        completion_time = []

        # questions = questions[:2]
        # ground_truths = ground_truths[:2]

        init_openai()
        for i, query in enumerate(questions):
            print(i, "begins")
            t0 = time.time()
            prompt, search_results = search_db(query, source)
            t1 = time.time()
            response = get_completion_openai(prompt, model=model)
            t2 = time.time()
            contexts.append(search_results) 
            answers.append(response)
            retrieve_time.append(t1 - t0)
            completion_time.append(t2 - t1)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "retrieve_time": retrieve_time,
            "completion_time": completion_time, 
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
        df_result.to_csv(f"./metrics/evaluation_chunk{chunk}_overlap{overlap}.csv", na_rep='NaN')