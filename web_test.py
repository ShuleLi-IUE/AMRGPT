import gradio as gr

js = """
console.log(123)
// 查找footer元素
var footerElement = document.querySelector("footer");
console.log("试图删除footer");
// 如果找到了footer元素，则删除它
if (footerElement) {
    footerElement.parentNode.removeChild(footerElement);
} else {
    console.log("未找到footer元素。");
}


"""

def UI1():
    with gr.Blocks(css="web_css.css") as demo:
        # gr.HTML("""<h1 align="center">Liuhui-bot</h1>
        #             <h3 align="center">for AMR policy</h3>
        #            """)

        
        with gr.Row() as output_field:
            with gr.Column() as chat_col:
                chatbot = gr.Chatbot(height=450, show_label=True, label="Chatbot")
            with gr.Column() as ref_col:
                # search_field = gr.Textbox(show_label=False, placeholder="Reference...", lines=14)
                search_field = gr.TextArea(show_label=True, label="Reference", placeholder="Reference...", elem_classes="box_height", container=False, lines=50)

        with gr.Column(elem_classes=".input_field") as input_field:
            with gr.Row(elem_classes=".dropdown_group"):
                model = gr.Dropdown(label="model", choices=["GPT-3.5", "GPT-4"], value=0, filterable=False, min_width=50)
                source = gr.Dropdown(label="source", choices=["Hybrid", "Only Database"], value=0, filterable=False)
                mode = gr.Dropdown(label="mode", choices=["Accuracy", "Efficiency"], value=0, filterable=False)
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Enter your questions about AMR...", lines=3)
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                emptyBtn = gr.Button("Clear")

        # gr.HTML("""<div class="at_bottom">Developed by Zhu Lab</div>""")
        context = gr.State([])


        # submitBtn.click(chat, [user_input, chatbot, context, search_field],
        #                 [user_input, chatbot, context, search_field])
        # emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8887, inbrowser=False, show_api=False, inline=True)


if __name__ == "__main__":
    UI1()