from openai_utils import get_embedding_openai
from embedding_utils import get_embedding_bge
import hnswlib
import numpy as np
import time

DIM = 1024
MAX_ELEMENTS = 2000000
THREADS = 4

class ShuleVectorDB:
    def __init__(self):
        self.index_hnsw = hnswlib.Index(space="l2", dim=DIM)
        self.index_hnsw.init_index(max_elements=MAX_ELEMENTS, ef_construction=100, M=32)
        self.index_hnsw.set_num_threads(THREADS)
        self.cnt = 0
        self.data_dict = {}
        '''
        "embeddings": ,
        "text": ,
        "type": paper, report
        "title": ,
        "year": ,
        "source": doi for paper, country for report,
        # "page": journal for paper, page for report
        '''
            
    def add_documents_dense(self, type, texts, title, year, source, verbose=True):
        n = len(texts)
        # ids = np.array([f"id_{i}" for i in np.arange(self.cnt, self.cnt + n)])
        ids = np.arange(self.cnt, self.cnt + n)
        t0 = time.time()
        embeddings=[get_embedding_bge(doc) for doc in texts]
        t1 = time.time()
        print("one embedding time ", (t1 - t0) / n)
        self.index_hnsw.add_items(embeddings, ids=ids)
        t2 = time.time()
        print("one add index time ", (t2 - t1) / n)
        # store documents to dict
        for i in range(n):
            self.data_dict[ids[i]] = {"embeddings": embeddings[i],
                                      "text": texts[i],
                                      "type": type,
                                      "title": title,
                                      "year": year,
                                      "source": source}
        self.cnt += n
        if verbose: 
            print(f"#{type}# Adding batch of {n} elements, now total index contains {self.cnt} elements")
            
    # def add_documents_sparse(self, type, texts, title, year, source, verbose=True):
            
    # def add_documents_hybrid(self, type, texts, title, year, source, verbose=True):
        
    def search_bge(self, query, top_n, verbose=True):
        t0 = time.time()
        embedding = get_embedding_bge(query)
        
        if verbose: 
            t1 = time.time()
            print("get_embedding_bge costs ", t1 - t0)
            
        labels, distances = self.index_hnsw.knn_query(embedding, k=top_n)
        
        if verbose: 
            t2 = time.time()
            print("index_hnsw.knn_query costs", t2 - t1)
        return labels[0].tolist()
    
    def get_context_by_labels(self, labels):
        print(type(labels), labels)
        print()
        texts = [self.data_dict[key]["text"] for key in labels]
        titles = [self.data_dict[key]["title"] for key in labels]
        years = [self.data_dict[key]["year"] for key in labels]
        sources = [self.data_dict[key]["source"] for key in labels]
        return texts, titles, years, sources
        
    
        
    
        