import openai
from openai import OpenAI
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
import time
from logger import log_info

 # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY
client = None
def init_openai():
    _ = load_dotenv(find_dotenv()) 
    openai.api_key = os.getenv('OPENAI_API_KEY')
    global client
    client = OpenAI()
    
def get_completion_openai(prompt, context=None, model="GPT-4", history="False"):
    """封装 openai 接口"""
    t0 = time.time()
    # messages = context + [{"role": "user", "content": prompt}]
    messages = (context + [{"role": "user", "content": prompt}]) if history == "True" else [{"role": "user", "content": prompt}]
    model_chosen = "gpt-4-turbo-preview" if model == "GPT-4" else "gpt-3.5-turbo"
    response = client.chat.completions.create(
        # model=model,
        model=model_chosen, 
        # "gpt-4"
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    i = 0
    while response.choices == None:
        i += 1
        if (i > 10): return "network failed"
        print(f"openai failed, retry in 5 seconds...\n\tresponse: {response}")
        time.sleep(5)
        response = client.chat.completions.create(
            # model=model,
            model=model_chosen, 
            # "gpt-4"
            messages=messages,
            temperature=0,  # 模型输出的随机性，0 表示随机性最小
        )
    log_info(f"get_completion_openai costs, {time.time() - t0}")
    return response.choices[0].message.content
    
def get_completion_openai_stream(prompt, context, model="GPT-4", history="True"):
    """封装 openai 接口"""
    t0 = time.time()
    messages = (context + [{"role": "user", "content": prompt}]) if history == "True" else [{"role": "user", "content": prompt}]
    print("gpt-4-turbo-preview" if model == "GPT-4" else "gpt-3.5-turbo") 
    response_stream = client.chat.completions.create(
        # model=model,
        # model="gpt-4-turbo-preview" if model == "GPT-4" else "gpt-3.5-turbo", 
        model="gpt-4-turbo-preview", 
        # "gpt-4"
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        stream=True
    )
    for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
        else:
            break
    log_info(f"get_completion_openai_stream costs, {time.time() - t0}")
    # return response.choices[0].message.content


# def get_embedding(text, model="text-embedding-ada-002"):
#     """封装 OpenAI 的 Embedding 模型接口"""
#     return client.embeddings.create(input=[text], model=model)['data'][0]['embedding']
#     return client.embeddings.create(input=[text], model=model)['data'][0]['embedding']

def get_embedding_openai(text, model="text-embedding-ada-002",dimensions=None):
    '''封装 OpenAI 的 Embedding 模型接口'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=text, model=model, dimensions=dimensions).data[0].embedding
    else:
        data = client.embeddings.create(input=text, model=model).data[0].embedding
    return data
    # return [x.embedding for x in data]
