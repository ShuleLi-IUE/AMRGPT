#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from text_utils import split_text

vec_db = InMemoryVecDB()

def init_db(file):
    print("---init database---")
    print("===extract_text_from_pdf===")
    paragraphs = extract_text_from_pdf(file.name)
    print("===split_text===")
    documents = split_text(paragraphs, 500, 100)
    print(len(documents))
    print(documents[:20])
    print("===2add_documents===")
    vec_db.add_documents(documents)
    print("db len", len(documents))
    print(documents[:20])


def chat(user_input, chatbot, context, search_field):
    print("---chat button---")
    print("===search in vectordb===")
    print("input: ", user_input)
    search_results = vec_db.search(user_input, 2)
    search_field = "\n\n".join(search_results)
    print("===building prompt===")
    prompt = build_prompt(info=search_results, query=user_input)
    print("prompt content built:\n", prompt)
    print("===get completion===")
    response = get_completion(prompt, context)
    print("completion content:\n", response)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field


def reset_state():
    print("---reset state---")
    return [], [], "", ""


def main():
    print("===begin gradio===")
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">AMRGPT</h1>""")

        with gr.Row():
            with gr.Column():
                fileCtrl = gr.File(label="上传文件", file_types=[',pdf'])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            with gr.Column(scale=2):
                # gr.HTML("""<h4>检索结果</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="检索结果...", lines=10)
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field],
                        [user_input, chatbot, context, search_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        fileCtrl.upload(init_db, inputs=[fileCtrl])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8889, inbrowser=True)


if __name__ == "__main__":
    main()
