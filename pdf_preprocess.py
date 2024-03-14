#!/usr/bin/env python
# coding=utf-8
# import gradio as gr
# from openai_utils import get_completion_openai
# from prompt_utils import build_prompt
from pdf_utils import  extract_text_from_pdf_pdfplumber_with_pages, extract_text_from_pdf_pdfminer
from text_utils import split_text, split_text_with_pages
# from rerank_utils import rerank
# from vectordb_utils_shule import ShuleVectorDB
# from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import os
import pandas as pd
import time
# import sys
# import errno
import pickle


corpus_path = "./corpus/AMR_GreyLiterature"

# report pdf
print("local database of report initing...") 
pdf_files = [file for file in os.listdir(corpus_path) if file.endswith('.pdf')]
pdf_files.sort()

for index, pdf_file in enumerate(pdf_files):    
    # if index in range(10): continue
    print(f"No. {index+1} pdf begin...")
    t0 = time.time()
    paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(os.path.join(corpus_path, pdf_file))
    # larger intersect
    documents, pages_matched = split_text_with_pages(paragraphs, pages, 400, 100)
    
    pd.DataFrame({"pages_matched": pages_matched,
                    "len": [len(doc) for doc in documents],
                    "documents": documents,}).to_csv("./corpus/preprocessed/" + str(index+1) + '. ' + pdf_file[:-4] + ".csv",
                                                    encoding="utf-8",
                                                    index=False)
                    
    # paragraphs = extract_text_from_pdf_pdfminer(os.path.join(corpus_path, pdf_file))
    # # larger intersect
    # documents = split_text(paragraphs, 400, 100)
    
    # pd.DataFrame({  "len": [len(doc) for doc in documents],
    #                 "documents": documents,}).to_csv("./corpus/preprocessed/" + str(index+1) + 'x. ' + pdf_file[:-4] + ".csv",
    #                                                 encoding="utf-8",
    #                                                 index=False)
    print(f"No. {index+1} pdf preprocess time: {time.time() - t0}")
