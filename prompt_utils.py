prompt_template = """
You're a question answering robot.
Your task is to answer the user's question based on the given given information below.
Make sure your response is based entirely on what you already know below. Don't make up answers.
If the following information is not sufficient to answer the user's question, please reply "I cannot answer your question".

Given information:
__INFO__

User's question：
__QUERY__

please answer user's question in user's language.
"""


def build_prompt(template=prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    prompt = template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt
