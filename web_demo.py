#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion_openai
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from text_utils import split_text
from rerank_utils import rerank
from vectordb_utils_shule import ShuleVectorDB
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd

vec_db = InMemoryVecDB()
vec_db_shule = ShuleVectorDB()
_ = load_dotenv(find_dotenv()) 
rerank_model = CrossEncoder(os.getenv('RERANK_MODEL_PATH'))
top_n = 5
recall_n = 50

def init_db_pdf(file):
    print("---init database---")
    paragraphs = extract_text_from_pdf(file.name)
    # larger intersect
    documents = split_text(paragraphs, 300, 150)

    print(len(documents))
    vec_db.add_documents_bge(documents)

# load data from ./corpus
def init_db_local():
    print("---init database begin---")
    corpus_path = os.getenv('CORPUS_PATH')
    
    # paper 
    print("local database of paper initing...") 
    df = pd.read_csv("/Users/lishule/Documents/namespace/AMRGPT/main/corpus/sample_8.csv")[["Title", "Year", "DOI", "Abstract"]]
    for i in range(df.shape[0]):
        documents = split_text(df["Abstract"][i], 400, 100)
        vec_db_shule.add_documents_bge(type="paper",
                                       texts=documents, 
                                       title=df["Abstract"][i],
                                       year=df["Year"][i],
                                       source=df["DOI"][i])
    
    # report pdf
    print("local database of report initing...") 
    pdf_files = [file for file in os.listdir(corpus_path) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        paragraphs = extract_text_from_pdf(os.path.join(corpus_path, pdf_file))
        # larger intersect
        documents = split_text(paragraphs, 400, 100)
        os.path.splitext(pdf_file)[0].split('_')
        vec_db_shule.add_documents_bge(type="report",
                                       texts=documents, 
                                       title=os.path.splitext(pdf_file)[0].split('_')[3],
                                       year=os.path.splitext(pdf_file)[0].split('_')[1],
                                       source=os.path.splitext(pdf_file)[0].split('_')[0])
    print("---init database finish---")
    

def chat(user_input, chatbot, context, search_field, search_strategy = "rerank"):
    print("---chat button---")
    search_results = []
    if search_strategy == "base":
        search_results = vec_db_shule.search_bge(user_input, top_n)
        search_field = "\n\n".join([f"{index+1}. {item}" for index, item in enumerate(search_results)])
    elif search_strategy == "rerank":
        print("===rerank===")
        search_results = rerank(user_input, top_n, recall_n)
        search_field = "\n\n".join([f"{index+1}. (score: {item[0]:.2e}), {item[1]}" for index, item in enumerate(search_results)])

    print("===building prompt===")
    prompt = build_prompt(info=[item[1] for item in search_results], query=user_input)
    print("prompt content built:\n", prompt)
    print("===get completion===")
    response = get_completion_openai(prompt, context)
    print("completion content:\n", response)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field

def rerank(user_input, top_n=5, recall_n=50, verbose=True):
    search_results = vec_db_shule.search_bge(user_input, recall_n)
    scores = rerank_model.predict([(user_input, doc) for doc in search_results])
    sorted_list = sorted(zip(scores,search_results), key=lambda x: x[0], reverse=True)
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

        with gr.Row():
            with gr.Column():
                fileCtrl = gr.File(label="Upload file", file_types=[',pdf'])

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

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8889, inbrowser=True)


if __name__ == "__main__":
    init_db_local()
    main()
