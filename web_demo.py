#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion_openai
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from text_utils import split_text
from rerank_utils import rerank
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
import os

vec_db = InMemoryVecDB()
_ = load_dotenv(find_dotenv()) 
path = os.getenv('RERANK_MODEL_PATH')
rerank_model = CrossEncoder(path)
top_n = 5
recall_n = 50

def init_db(file):
    print("---init database---")
    paragraphs = extract_text_from_pdf(file.name)
    # larger intersect
    documents = split_text(paragraphs, 300, 150)
    print(len(documents))
    vec_db.add_documents_bge(documents)


def chat(user_input, chatbot, context, search_field, search_strategy = "rerank"):
    print("---chat button---")
    search_results = []
    if search_strategy == "base":
        search_results = vec_db.search_bge(user_input, top_n)
        search_field = "\n\n".join([f"{index+1}. {item}" for index, item in enumerate(search_results)])
    elif search_strategy == "rerank":
        search_results = rerank(user_input, top_n, recall_n)
        search_field = "\n\n".join([f"{index+1}. (score: {item[0]:.2e}), {item[1]}" for index, item in enumerate(search_results)])
    print("===building prompt===")
    prompt = build_prompt(info=search_results, query=user_input)
    print("prompt content built:\n", prompt)
    print("===get completion===")
    response = get_completion_openai(prompt, context)
    print("completion content:\n", response)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field

def rerank(user_input, top_n = 5, recall_n = 50):
    search_results = vec_db.search_bge(user_input, recall_n)
    scores = rerank_model.predict([(user_input, doc) for doc in search_results])
    sorted_list = sorted(zip(scores,search_results), key=lambda x: x[0], reverse=True)
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

        fileCtrl.upload(init_db, inputs=[fileCtrl])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8889, inbrowser=True)


if __name__ == "__main__":
    main()
