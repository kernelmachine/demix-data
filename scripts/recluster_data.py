import argparse
import gzip
import itertools
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS
from domain_loader.domain_loader import Domain, collate_fn
from domain_loader.utils import take_n_tokens
from fairseq.data import data_utils
from scipy import sparse
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm
from sklearn.decomposition import TruncatedSVD
import json



def sample_docs(domain, sample=None, sample_from_head=False):
    if domain in ['reddit', '1b']:
        if sample:
            sample_ = 1
        else:
            sample_ = None
    else:
        if sample:
            sample_ = sample
        else:
            sample_ = None
    dataset = Domain(PROJECT_DIR / domain / domain, sample=sample_)
    

    loader = DataLoader(dataset,
                        num_workers=0,
                        batch_size=16,
                        collate_fn=collate_fn)
    
    pbar = tqdm(loader)
    ms = []
    vs = []
    
    docs = []
    if domain in ['reddit', '1b']:
        for fname,text, _ in pbar:
            for f,t in zip(fname, text):
                for tt in t.split('<|endoftext|>'):
                    docs.append({"fname": f, "text": tt})
    else:
        for fname, text, _ in pbar:
            for f,t in zip(fname, text):
                docs.append({"fname": f, "text": t})
    if sample and len(docs) > sample:
        docs = random.sample(docs, sample)
    
    return docs


def get_clusters(domain, svd, count_vectorizer, kmeans, sample,  sample_from_head=False):
    eval_docs = sample_docs(domain, sample, sample_from_head=sample_from_head)
    
    eval_embeddings = count_vectorizer.transform(tqdm([x['text'] for x in eval_docs]))
    eval_vectors = svd.transform(eval_embeddings).astype(np.float32)
    
    _, I = kmeans.index.search(eval_vectors, 1)
    I = I.squeeze(1)
    for cluster_id, item in zip(I, eval_docs):
        item['cluster'] = int(cluster_id)
    return eval_docs

if __name__ == '__main__':
    domains = ['1b', 'cs', 'legal', 'med', 'openwebtext', 'realnews', 'reddit', 'reviews']
    docs = []
    dim = 64
    kmeans = faiss.Kmeans(dim, 8, niter=60, verbose=True, gpu=True)
    for domain in domains:
        ds = sample_docs(domain, sample=1000, sample_from_head=True)
        docs.extend(ds)
    count_vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = count_vectorizer.fit_transform(tqdm([x['text'] for x in docs]))
    svd = TruncatedSVD(n_components=64)
    vectors = svd.fit_transform(embeddings).astype(np.float32)
    kmeans.train(vectors)

    from fairseq import pdb; pdb.set_trace()
    dfs = []
    with open('reclustered_docs.jsonl', 'w+') as f:
        for domain in tqdm(domains):
            clusters = get_clusters(domain, svd, count_vectorizer, kmeans, None, sample_from_head=False)
            for item in clusters:
                item['domain_label'] = domain
                del item['text']
                f.write(json.dumps(item) + '\n')
