import argparse
import gzip
import itertools
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import pickle
# import faiss
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
from typing import TypeVar, Iterable, List, Sequence, Union, Any
from sklearn.cluster import MiniBatchKMeans
from k_means_constrained import KMeansConstrained

T = TypeVar('T')


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def sample_docs(domain, sample=None, sample_from_head=True):
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
    dataset = Domain(PROJECT_DIR / domain / domain, sample=sample_, sample_from_head=True)
    
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


def get_clusters(domain, svd, count_vectorizer, kmeans, sample, balanced=False, preloaded=False, batched_write=False, output_dir=None):
    dataset = Domain(PROJECT_DIR / domain / domain, sample=sample, sample_from_head=True)
    if domain == '1b':
        loader = DataLoader(dataset, num_workers=0, batch_size=100, collate_fn=collate_fn)
    elif domain == 'reddit':
        loader = DataLoader(dataset, num_workers=0, batch_size=1, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, num_workers=0, batch_size=10000, collate_fn=collate_fn)
    pbar = tqdm(loader)
    pbar.set_description(f"{domain}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        for i in range(8):
            (output_dir / str(i)).mkdir()
    for fname,text, _ in pbar:
        docs = []
        if domain in ['reddit', '1b']:
            for t in text:
                for tt in t.split('<|endoftext|>'):
                    docs.append({"text": tt})
        else:
            for t in text:
                docs.append(t)
        eval_embeddings = count_vectorizer.transform(tqdm(docs))
        eval_vectors = svd.transform(eval_embeddings).astype(np.float32)
        if balanced:
            I = kmeans.predict(eval_vectors).tolist()
        else:
            _, I = kmeans.search(eval_vectors, 1)
            I = I.squeeze(1)
        z = defaultdict(list)
        for cluster_id, doc in zip(I, docs):
            z[cluster_id].append("<|endoftext|> " + doc.strip())
        for cluster_id, docs in z.items():
            with open(Path(output_dir) / str(cluster_id) / f'train.{domain}.txt', 'a+') as f:
                for doc in docs:
                    f.write(doc + "\n")
          

    #eval_docs = sample_docs(domain, sample)
    #if batched_write:
    #    with open(output_file, 'w+') as f:
    #        eval_docs_iter = tqdm(batchify(eval_docs, batch_size=1000000))
    #        for eval_docs in eval_docs_iter:
    #            eval_embeddings = count_vectorizer.transform(tqdm([x['text'] for x in eval_docs]))
    #            eval_vectors = svd.transform(eval_embeddings).astype(np.float32)
#
#                if preloaded:
#                    _, I = kmeans.search(eval_vectors, 1)
#                else:
#                    _, I = kmeans.index.search(eval_vectors, 1)
#                I = I.squeeze(1)
#                for cluster_id, item in zip(I, eval_docs):
#                    item['cluster'] = int(cluster_id)
#                for item in eval_docs:
#                    item['domain_label'] = domain
#                    del item['text']
#                    f.write(json.dumps(item) + '\n')
#    else:
#        with open(output_file, 'w+') as f:
#            eval_embeddings = count_vectorizer.transform(tqdm([x['text'] for x in eval_docs]))
#            eval_vectors = svd.transform(eval_embeddings).astype(np.float32)
#            if preloaded:
#                _, I = kmeans.search(eval_vectors, 1)
#            else:
#                _, I = kmeans.index.search(eval_vectors, 1)
#            I = I.squeeze(1)
#            for cluster_id, item in zip(I, eval_docs):
#                item['cluster'] = int(cluster_id)
#            for item in eval_docs:
#                item['domain_label'] = domain
#                del item['text']
#                f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    domains = ['1b', 'cs', 'reddit', 'realnews', 'openwebtext', 'reviews', 'legal', 'med'] 
    docs = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--kmeans_output_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--domains', nargs="+", type=str)
    parser.add_argument('--min_cluster_size', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()
    if args.domains:
        domains = args.domains
    if not args.load:
        args.kmeans_output_dir.mkdir(exist_ok=True)
        #if args.balanced:
            #kmeans = KMeansConstrained(n_clusters=args.num_clusters,
             #                          init=domain_embeddings,
              #                          size_min=args.min_cluster_size)
        #else:
         #   kmeans = faiss.Kmeans(args.dim, args.num_clusters, verbose=True, gpu=True)
        domain_ids = []
        for domain_id,domain in enumerate(domains):
            ds = sample_docs(domain, sample=args.sample_size)
            docs.extend(ds)
            domain_ids.extend([domain_id]*len(ds))
        count_vectorizer = TfidfVectorizer(stop_words="english")
        embeddings = count_vectorizer.fit_transform(tqdm([x['text'] for x in docs]))
        svd = TruncatedSVD(n_components=args.dim)
        vectors = svd.fit_transform(embeddings).astype(np.float32)
        domain_ids = np.array(domain_ids)
        domain_embeddings = []
        for domain_id in np.unique(domain_ids):
            domain_embedding = np.mean(vectors[domain_ids == domain_id,:], 0)
            domain_embeddings.append(np.expand_dims(domain_embedding,1))
        domain_embeddings = np.concatenate(domain_embeddings,1)
        kmeans = MiniBatchKMeans(n_clusters=args.num_clusters, init=domain_embeddings.T)
        if args.balanced:
            out = kmeans.fit_predict(vectors)
            with open(args.kmeans_output_dir / "kmeans.index", "wb+") as f:
                pickle.dump(kmeans, f)
        else:
            kmeans.train(vectors)
            np.save(args.kmeans_output_dir / "kmeans.index", kmeans.centroids)

        with open(args.kmeans_output_dir / "tfidf.pkl", "wb+") as f:
            pickle.dump(count_vectorizer, f)
        with open(args.kmeans_output_dir / "svd.pkl", "wb+") as f:
            pickle.dump(svd, f)

    else:
        with open(args.load / "tfidf.pkl", "rb") as f:
            count_vectorizer = pickle.load(f)
        if args.balanced:
            with open(args.load / 'kmeans.index', 'rb') as f:
                kmeans = pickle.load(f)   
        else:   
            centroids = np.load(args.load / "kmeans.index.npy")
            kmeans = faiss.IndexFlatL2(args.dim)
            kmeans.add(centroids)
        with open(args.load / 'svd.pkl', 'rb') as f:
            svd = pickle.load(f)        
 
    dfs = []
    for domain in tqdm(domains):
        get_clusters(domain,
                    svd,
                    count_vectorizer,
                    kmeans,
                    sample=None,
                    preloaded=args.load or False,
                    batched_write=domain == 'reddit',
                    output_dir=args.output_dir,
                    balanced=args.balanced)

