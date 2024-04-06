#!/usr/bin/env python
# coding=utf-8
from vectordb_utils_shule import ShuleVectorDB
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
import os
import pandas as pd
import time
import sys
import errno
import pickle
import re
# import warnings
# warnings.filterwarnings("ignore")

distance = "l2"
batch_size = 12
vec_db_shule = ShuleVectorDB(space=distance,
                             batch_size=batch_size)
# def init_db_pdf(file):
#     print("---init database---")
#     paragraphs, pages = extract_text_from_pdf_pdfplumber_with_pages(file.name)
#     # larger intersect
#     documents = split_text_with_pages(paragraphs, pages, 300, 150)
#     print(len(documents))
#     vec_db.add_documents_bge(documents)

# load data from ./corpus
def init_db_local(chunk_size, overlap):
    print(f"---chunk: {chunk_size}, overlap: {overlap} begin---")
    corpus_path = f"./corpus/preprocessed_chunk{chunk_size}_overlap{overlap}"
        
    # report pdf
    # print("local database of report initing...") 
    pdf_files = [file for file in os.listdir(corpus_path) if file.endswith('.csv')]
    pdf_files.sort()
    for file_name in pdf_files:
        t0 = time.time()
        match = re.match(r"^(\d+)\.\s(.+)\.csv", file_name)
        if match:
            data = pd.read_csv(corpus_path + "/" + file_name)
            i = match.group(1)
            # if (int(i) > 2): continue
            print(f"{file_name} begin")
            name = match.group(2)
            context = os.path.splitext(name)[0].split('_')
            print(context)
            vec_db_shule.add_documents_dense(type="report",
                                        texts=data['documents'].tolist(), 
                                        pages=data['pages_matched'].to_list(),
                                        title=context[4],
                                        year=context[2],
                                        country=context[1],
                                        ORG=context[0])
        print(f"{file_name} end, cost time: {time.time()-t0}")
        
    # print("---init database end---")
    # print("---ShuleVectorDB dump begin---")
    t2 = time.time()
    file_name = f"./evaluation/chunks/index_chunk{chunk_size}_overlap{overlap}.pickle"
    vec_db_shule.dump(file_name)
    # print("---ShuleVectorDB dump end---")
    # print(f"dump costs time: {time.time()-t2}")

if __name__ == "__main__":
    overlap_rate = 0
    for chunk_size in [256, 512, 1024]:
        # for overlap_rate in [0, 0.1, 0.2, 0.4]:
        init_db_local(chunk_size, overlap_rate)
