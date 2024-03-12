#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion_openai
from prompt_utils import build_prompt
from pdf_utils import  extract_text_from_pdf_pdfplumber_with_pages
from text_utils import split_text, split_text_with_pages
from rerank_utils import rerank
from vectordb_utils_shule import ShuleVectorDB
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

top_n = 5
recall_n = 30
distance = "l2"
batch_size = 12

vec_db_shule = ShuleVectorDB(space=distance,
                             batch_size=batch_size)
_ = load_dotenv(find_dotenv()) 
rerank_model = CrossEncoder(os.getenv('RERANK_MODEL_PATH'))


# def init_db_pdf(file):
#     print("---init database---")
#     paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(file.name)
#     # larger intersect
#     documents = split_text_with_pages(paragraphs, pages, 300, 150)
#     print(len(documents))
#     vec_db.add_documents_bge(documents)

# load data from ./corpus
def init_db_local():
    print("---init database begin---")
    corpus_path = os.getenv('CORPUS_PATH')
    
    # # paper 
    # print("local database of paper initing...") 
    # df = pd.read_csv("/Users/lishule/Documents/namespace/AMRGPT/main/corpus/sample_8.csv")[["Title", "Year", "DOI", "Abstract", "Journal"]]
    # for i in range(df.shape[0]):
    #     documents = split_text(df["Abstract"][i], 400, 100)
    #     vec_db_shule.add_documents_bge(type="paper",
    #                                    texts=documents, 
    #                                    title=df["Abstract"][i],
    #                                    year=df["Year"][i],
    #                                    source=df["DOI"][i])
    
    # report pdf
    print("local database of report initing...") 
    pdf_files = [file for file in os.listdir(corpus_path) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(os.path.join(corpus_path, pdf_file))
        # larger intersect
        documents, pages_matched = split_text_with_pages(paragraphs, pages, 400, 100)
        context = os.path.splitext(pdf_file)[0].split('_')
        pd.DataFrame({"documents": documents,
                     "pages_matched": pages_matched}).to_csv("tmp.csv")
        vec_db_shule.add_documents_dense(type="report",
                                       texts=documents, 
                                       pages=pages_matched,
                                       title=context[4],
                                       year=context[2],
                                       country=context[1],
                                       ORG=context[0])
    print("---init database finish---")
 
    
# search = (hnsw, rerank, fusion)
def chat(user_input, chatbot, context, search_field, search_strategy = "rerank"):
    print("---chat button---")
    search_results = []
    
    if search_strategy == "hnsw":
        search_labels = vec_db_shule.search_bge(user_input, top_n)
        texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(info=[f"{texts[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        
    elif search_strategy == "rerank":
        print("===rerank===")
        search_results = rerank(user_input, top_n, recall_n)
        search_field = "\n\n".join([f"{index+1}. [Reference: {item[2]}, Page: {item[5]}, ORG: {item[6]}, Year: {item[3]}]\n(Score: {item[0]:.2e}) {item[1]}" for index, item in enumerate(search_results)])
        prompt = build_prompt(info=[f"{item[1]} [Reference: Page {item[5]}, {item[2]}, {item[3]}, {item[6]}]" for item in search_results], query=user_input)
        
    elif search_strategy == "fusion":
        print("Not support yet.")
            
    print("prompt content built:\n", prompt)
    
    print("===get completion===")
    t0 = time.time()
    response = get_completion_openai(prompt, context)
    print("get_completion_openai costs", time.time() - t0)
    print("completion content:\n", response)
    
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field

def rerank(user_input, top_n=5, recall_n=30, verbose=True):
    search_labels = vec_db_shule.search_bge(user_input, recall_n)
    t0 = time.time()
    texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
    t1 = time.time()
    if verbose: print("vec_db_shule.get_context_by_labels costs: ", t1 - t0)
    scores = rerank_model.predict([(user_input, doc) for doc in texts])
    t2 = time.time()
    if verbose: print("rerank_model.predict costs: ", t2 - t0)
    print()
    sorted_list = sorted(zip(scores, texts, titles, years, countries, pages, ORGs), key=lambda x: x[0], reverse=True)
    if verbose:
        print(f"finish rerank {len(sorted_list)} texts, return highest {top_n} texts")
    # for score, doc in sorted_list:
    #     print(f"{score}\t{doc}\n")
    return sorted_list[:top_n]
    # return [item[1] for item in sorted_list[:top_n]]


def reset_state():
    print("---reset state---")
    return [], [], "", ""


def main():
    print("===begin gradio===")
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">AMRGPT</h1>
                   <h3 align="center">Zhu Lab</h3>""")

        # with gr.Row():
        #     with gr.Column():
        #         fileCtrl = gr.File(label="Upload file", file_types=[',pdf'])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height = 520)
            with gr.Column(scale=2):
                              
                user_input = gr.Textbox(show_label=False, placeholder="Enter your questions about AMR...", lines=3)
                with gr.Row():
                    submitBtn = gr.Button("Submit", variant="primary")
                    emptyBtn = gr.Button("Clear")
                search_field = gr.Textbox(show_label=False, placeholder="Reference...", lines=14)

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field],
                        [user_input, chatbot, context, search_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        # fileCtrl.upload(init_db_pdf, inputs=[fileCtrl])

    demo.queue().launch(share=True, server_name='0.0.0.0', server_port=8889, inbrowser=True)


if __name__ == "__main__":
    init_db_local()
    main()