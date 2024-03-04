import openai
from openai import OpenAI
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv

 # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

# openai.api_key = os.getenv('OPENAI_API_KEY')
_ = load_dotenv(find_dotenv()) 
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
    
def get_completion(prompt, context, model="gpt-3.5-turbo"):
    """封装 openai 接口"""
    messages = context + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


# def get_embedding(text, model="text-embedding-ada-002"):
#     """封装 OpenAI 的 Embedding 模型接口"""
#     return client.embeddings.create(input=[text], model=model)['data'][0]['embedding']
#     return client.embeddings.create(input=[text], model=model)['data'][0]['embedding']

def get_embedding(text, model="text-embedding-ada-002",dimensions=None):
    '''封装 OpenAI 的 Embedding 模型接口'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=text, model=model, dimensions=dimensions).data[0].embedding
    else:
        data = client.embeddings.create(input=text, model=model).data[0].embedding
    return data
    # return [x.embedding for x in data]
