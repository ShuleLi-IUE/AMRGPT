#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion_openai, init_openai, get_completion_openai_stream
from prompt_utils import build_prompt
from pdf_utils import  extract_text_from_pdf_pdfplumber_with_pages
from text_utils import split_text, split_text_with_pages
from vectordb_utils_shule import ShuleVectorDB
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import os
import pandas as pd
import time
import sys
import errno
import pickle
from logger import log_info, log_debug, log_warning
# import warnings
# warnings.filterwarnings("ignore")

top_n = 8
recall_n = 80
distance = "l2"
batch_size = 12
num_workers = None
search_strategy = "rerank"
vec_db_shule = ShuleVectorDB(space=distance,
                             batch_size=batch_size)
rerank_model = CrossEncoder(os.getenv('RERANK_MODEL_PATH')) if search_strategy == "rerank" else None
# def init_db_pdf(file):
#     print("---init database---")
#     paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(file.name)
#     # larger intersect
#     documents = split_text_with_pages(paragraphs, pages, 300, 150)
#     print(len(documents))
#     vec_db.add_documents_bge(documents)

# load data from ./corpus
def init_db_local():
    log_info("---init local database begin---")
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
    log_info("local database of report initing...") 
    pdf_files = [file for file in os.listdir(corpus_path) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(os.path.join(corpus_path, pdf_file))
        # larger intersect
        documents, pages_matched = split_text_with_pages(paragraphs, pages, 400, 100)
        context = os.path.splitext(pdf_file)[0].split('_')
        # pd.DataFrame({"documents": documents,
        #              "pages_matched": pages_matched}).to_csv("tmp.csv")
        vec_db_shule.add_documents_dense(type="report",
                                       texts=documents, 
                                       pages=pages_matched,
                                       title=context[4],
                                       year=context[2],
                                       country=context[1],
                                       ORG=context[0])
    log_info("---init database end---")
    vec_db_shule.dump()
    log_info("---ShuleVectorDB dump end---")

def init_db_load_index(index_path):
    global vec_db_shule
    init_openai()
    log_info("---init database by index file begin---")
    log_info(f"index path: {index_path}")
    with open(index_path, 'rb') as file:
        vec_db_shule = pickle.load(file)
    
    log_info("---init database by index file end---")
    
# search = (hnsw, rerank, fusion)
def search_db(user_input, chatbot, context, search_field, source_type, mode_type):
    log_info("---chat button---")
    
    if search_strategy == "hnsw" or mode_type == "Efficiency":
        log_info("===hnsw===")
        search_labels = vec_db_shule.search_bge(user_input, top_n)
        texts, pages, titles, years, countries, ORGs = vec_db_shule.get_context_by_labels(search_labels)
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        
    elif search_strategy == "rerank" or mode_type == "Accuracy":
        log_info("===rerank===")
        scores, texts, pages, titles, years, countries, ORGs = rerank(user_input, top_n, recall_n)
        res = [texts[i] if countries[i] == 'xxx' else f'In {countries[i]}, {texts[i]}' for i in range(top_n)]

        search_field = "\n\n".join([f"{i+1}. [Reference: {titles[i]}, Page: {pages[i]}, ORG: {ORGs[i]}, Year: {years[i]}]\n{texts[i]}" for i in range(top_n)])
        prompt = build_prompt(source_type=source_type, info=[f"{res[i]} [Reference: Page {pages[i]}, {titles[i]}, {years[i]}, {ORGs[i]}]" for i in range(top_n)], query=user_input)
        
    elif search_strategy == "fusion":
        log_warning("Not support yet.")
        return
            
    log_info(f"prompt content built:\n{prompt}")
    return prompt, search_field


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

    # sorted_list = {'scores': scores, 
    #                 'texts': [texts[i] for i in ids], 
    #                 'titles':[titles[i] for i in ids], 
    #                 'years':[years[i] for i in ids], 
    #                 'countries':[countries[i] for i in ids], 
    #                 'pages':[pages[i] for i in ids], 
    #                 'ORGs':[ORGs[i] for i in ids]}
    log_info(f"finish rerank {recall_n} texts, return highest {top_n} texts")
    # for score, doc in sorted_list:
    #     print(f"{score}\t{doc}\n")
    return scores, [texts[i] for i in ids], [pages[i] for i in ids], [titles[i] for i in ids], [years[i] for i in ids], [countries[i] for i in ids], [ORGs[i] for i in ids]
    # return [item[1] for item in sorted_list[:top_n]]


def reset_state():
    log_info("---reset state---")
    return [], [], "", ""


def main():
    log_info("===begin gradio===")
    with gr.Blocks(css="web_css.css") as demo:
        gr.HTML("""<h1 align="center">Liuhui-bot</h1>
                    <h3 align="center">for AMR policy</h3>
                   """)

        
        with gr.Row() as output_field:
            with gr.Column() as chat_col:
                chatbot = gr.Chatbot(height=450, show_label=True, label="Chatbot")
            with gr.Column() as ref_col:
                # search_field = gr.Textbox(show_label=False, placeholder="Reference...", lines=14)
                search_field = gr.TextArea(show_label=True, label="Reference", placeholder="Reference...", elem_classes="box_height", container=False, lines=50)

        with gr.Column(elem_classes=".input_field") as input_field:
            with gr.Row(elem_classes=".dropdown_group"):
                model = gr.Dropdown(label="Model", choices=["GPT-3.5", "GPT-4"], value="GPT-4", filterable=False, min_width=50)
                source = gr.Dropdown(label="Source", choices=["Hybrid", "Standalone"], value="Hybrid", filterable=False)
                mode = gr.Dropdown(label="Mode", choices=["Accuracy", "Efficiency"], value="Accuracy", filterable=False)
                history = gr.Dropdown(label="History", choices=["Yes", "No"], value="Yes", filterable=False)
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Enter your questions about AMR...", lines=3)
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                emptyBtn = gr.Button("Clear")

        gr.HTML("""<div class="at_bottom">Developed by Zhu Lab</div>""")
        context = gr.State([])

        def user(user_message, history):
            return user_message, history + [[user_message, None]]
        
        def bot2(user_input, chatbot, context, search_field, model, source, mode, history):
            print(model, source, mode, history, user_input)
            log_info(f"model: {model}, source: {source}, mode: {mode}, history: {history}\nuser_input:{user_input}")
            prompt, search_field = search_db(user_input, chatbot, context, search_field, source, mode)
            
            # clear user input
            user_input = ""

            # print("prompt and search_field:", prompt, search_field)
            log_info("===get completion===")
            response_stream = get_completion_openai_stream(prompt, context, model, history)
            response = ""
            chatbot[-1][1] = ""
            for word in response_stream:
                chatbot[-1][1] += word
                response += word
                yield user_input, chatbot, context, ""
            
            context.append({'role': 'user', 'content': user_input})
            context.append({'role': 'assistant', 'content': response})
            log_info(f"completion content:\n{response}")

            # response is empty, lead to a error in gradio
            # put a netword error to client
            if chatbot[-1][1] == "":
                chatbot[-1][1] += "Network Error. Wait seconds and try again."
                search_field = ""
            
            yield user_input, chatbot, context, search_field

        submitBtn.click(user, [user_input, chatbot],
                        [user_input, chatbot], queue=False
                        ).then(
                            bot2, 
                            [user_input, chatbot, context, search_field, model, source, mode, history], 
                            [user_input, chatbot, context, search_field]
                        )
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        # fileCtrl.upload(init_db_pdf, inputs=[fileCtrl])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8889, inbrowser=False, show_api=False)

def init():
    index_path = sys.argv[1] if len(sys.argv) > 1 else None
    if index_path == None:
        init_db_local()
    elif os.path.exists(index_path):
        init_db_load_index(index_path)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), index_path)
    
if __name__ == "__main__":
    init()
    main()