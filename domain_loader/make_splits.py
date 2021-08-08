import argparse
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import humanize
import numpy as np
import pandas as pd
import torch
import re
from sklearn.cluster import MiniBatchKMeans
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from transformers import GPT2Tokenizer
from typing import TypeVar, Iterable, List, Sequence, Union, Any


from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS
from domain_loader.domain_loader import Domain
from domain_loader.utils import take_n_tokens

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


def load_text(domain, add_bos_token, num_workers, batch_size, num_expected_tokens=None, num_expected_docs=None):
    with open(PROJECT_DIR / domain  / "splits-final" / "train_files.txt", 'r') as f:
        files = [x.strip() for x in tqdm(f.readlines())]
        np.random.shuffle(files)

    dataset = Domain(PROJECT_DIR / domain / domain,
                     filenames=files if domain not in ['1b', 'reddit'] else None,
                     add_bos_token=add_bos_token,
                     track_token_count=True)

    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    pbar = tqdm(loader)
    texts = []

    curr_tokens = 0
    curr_docs = 0
    written = False
    for _, text ,token_count, _ in pbar:
        s = f"{domain}, num tokens: {humanize.intword(curr_tokens)}, num docs: {humanize.intword(curr_docs)}"
        pbar.set_description(s)
        if (num_expected_docs and curr_docs > num_expected_docs):
            texts = texts[:num_expected_docs]
            break
        if domain in ['1b', 'reddit']:
            tt = [t.split("<|endoftext|>") for t in text]
            text = [y for x in tt for y in x]
            token_count = [len(x.split()) for x in text]
        if (num_expected_tokens and curr_tokens > num_expected_tokens):
            if not written:
                text = " ".join(text)[:num_expected_tokens]
                texts.extend(text)
            else:
                texts = "\n".join(texts)[:num_expected_tokens]
                texts = texts.split('\n')
                curr_tokens = num_expected_tokens
            break
        else:
            texts.extend(text)
            curr_tokens += sum(token_count)
            curr_docs += len(text)
        written = True
    return texts, curr_tokens, curr_docs

def extract_features(domain, add_bos_token, num_workers, batch_size, num_expected_tokens=100000, clusterer=None):
    texts, curr_tokens, curr_docs = load_text(domain, add_bos_token, num_workers, batch_size, num_expected_tokens)
    print(f'retrieved {curr_tokens} tokens, {curr_docs} docs')
    vecs = clusterer['vectorizer'].fit_transform(tqdm(texts))
    vecs = clusterer['svd'].fit_transform(vecs)
    return vecs


def get_cluster_id(clusterer, text):
    text = [x.replace("<|endoftext|>", "") for x in text]
    vec = clusterer['vectorizer'].transform(text)
    vec = clusterer['svd'].transform(vec)
    cluster_id = clusterer['kmeans'].predict(vec)
    return cluster_id


def write_split(domain, add_bos_token, num_workers, batch_size, output_dir, split, files=None, ignore_files=[], clusterer=None, from_file=None, anonymize=False):

    if not from_file:
        dataset = Domain(PROJECT_DIR / domain / domain, filenames=files, add_bos_token=add_bos_token, ignore_files=ignore_files, anonymize=anonymize)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    else:
        fh = open(from_file, 'r')
        loader = batchify(fh.read().split('<|endoftext|>'), 10000)
        fh.close()
    files = []
    done = False

    if clusterer:
        print("Detected clusterer. Clustering while loading splits.")
        filehandle = [open(output_dir / f"{split}.{i}.txt", "w+") for i in range(clusterer['kmeans'].n_clusters)]
        curr_tokens = dict((i, 0) for i in range(clusterer['kmeans'].n_clusters))
    else:
        filehandle = [open(output_dir / f"{split}.txt", "w+")]
        curr_tokens = {0: 0}

    pbar = tqdm(loader)
    written = False






    if not from_file:
        for fname, text,_, _ in pbar:
            files.extend(fname)
            if clusterer:
                if domain in ['1b', 'reddit']:
                    tt = [t.split("<|endoftext|>") for t in text]
                    text = ["<|endoftext|> " + y for x in tt for y in x]
                cluster_ids = get_cluster_id(clusterer, text)
                iter_ = zip(cluster_ids, text)
            else:
                iter_ = text
            for item in iter_:
                if clusterer:
                    cluster_id = item[0]
                    doc = item[1]
                else:
                    doc = item
                if not doc or doc == "<|endoftext|> ":
                    continue
                if clusterer:
                    s = f"{split}, "
                    for i, tok in curr_tokens.items():
                        s += f"cluster {i}: {humanize.intword(tok)} || "
                else:
                    s = f"{split}, num tokens: {humanize.intword(curr_tokens[0])}"
                pbar.set_description(s)
                if sum(curr_tokens.values()) > TOKEN_COUNTS[domain][f'num_{split}_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(doc.split()[:TOKEN_COUNTS[domain][f'num_{split}_tokens']])
                        if clusterer:
                            filehandle[cluster_id].write(doc.strip() + "\n")
                            curr_tokens[cluster_id] += len(doc.split())
                        else:
                            filehandle[0].write(doc.strip() + "\n")
                            curr_tokens[0] += len(doc.split())
                        written = True
                    done = True
                    break
                if clusterer:
                    filehandle[cluster_id].write(doc.strip() + "\n")
                    curr_tokens[cluster_id] += len(doc.split())
                else:
                    filehandle[0].write(doc.strip() + "\n")
                    curr_tokens[0] += len(doc.split())
                written = True
            if done:
                break
    else:
        for text in pbar:
            text = ["<|endoftext|> " + x for x in text]
            if clusterer:
                cluster_ids = get_cluster_id(clusterer, text)
                iter_ = zip(cluster_ids, text)
            else:
                iter_ = text
            for item in iter_:
                if clusterer:
                    cluster_id = item[0]
                    doc = item[1]
                else:
                    doc = item
                if not doc:
                    continue
                if clusterer:
                    s = f"{split}, "
                    for i, tok in curr_tokens.items():
                        s += f"cluster {i}: {humanize.intword(tok)} || "
                else:
                    s = f"{split}, num tokens: {humanize.intword(curr_tokens[0])}"
                pbar.set_description(s)
                if sum(curr_tokens.values()) > TOKEN_COUNTS[domain][f'num_{split}_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(doc.split()[:TOKEN_COUNTS[domain][f'num_{split}_tokens']])
                        if clusterer:
                            filehandle[cluster_id].write(doc.strip() + "\n")
                            curr_tokens[cluster_id] += len(doc.split())
                        else:
                            filehandle[0].write(doc.strip() + "\n")
                            curr_tokens[0] += len(doc.split())
                        written = True
                    done = True
                    break
                if clusterer:
                    filehandle[cluster_id].write(doc.strip() + "\n")
                    curr_tokens[cluster_id] += len(doc.split())
                else:
                    filehandle[0].write(doc.strip() + "\n")
                    curr_tokens[0] += len(doc.split())
                written = True
            if done:
                break
    for fh in filehandle:
        fh.close()
    if from_file:
        return None, None, curr_tokens
    else:
        return dataset.files, files, curr_tokens

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--add-bos-token", action='store_true')
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--train-files", type=Path, default=None)
    parser.add_argument("--dev-files", type=Path, default=None)
    parser.add_argument("--test-files", type=Path, default=None)
    parser.add_argument("--pretrain-clusters-only", action='store_true')
    parser.add_argument("--pretrain-clusters", nargs="+", type=str)
    parser.add_argument("--output-clusters", type=Path)
    parser.add_argument("--from-file", type=Path)
    parser.add_argument("--train-only", action='store_true')
    parser.add_argument("--dev-only", action='store_true')
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--anonymize", action='store_true')

    args = parser.parse_args()
    domain = args.domain

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True)

    if args.pretrain_clusters:
        clusterer = {"svd": TruncatedSVD(n_components=64),
                     "vectorizer": TfidfVectorizer(stop_words="english")}

        texts = {}
        # load texts
        for domain in args.pretrain_clusters:
            if domain in ['reddit', '1b']:
                add_bos_token = False
                num_workers = 1
                batch_size = 1
            else:
                add_bos_token = False
                num_workers = 1
                batch_size = 16
            text, curr_tokens, curr_docs = load_text(domain, add_bos_token, num_workers, batch_size, num_expected_docs=100000)

            texts[domain] = {'text': text}


        ts = [texts[domain]['text'] for domain in texts]
        ts = [y for x in ts for y in x]
        vecs = clusterer['vectorizer'].fit_transform(ts)
        clusterer['svd'].fit(vecs)
        domain_embeddings = []
        for domain in texts:
            texts[domain]['vecs'] = clusterer['vectorizer'].transform(texts[domain]['text'])
            texts[domain]['vecs'] = clusterer['svd'].transform(texts[domain]['vecs'])
            domain_embeddings.append(np.expand_dims(np.mean(texts[domain]['vecs'], 0),1))
        domain_embeddings = np.concatenate(domain_embeddings, 1)
        kmeans = KMeansConstrained(n_clusters=8, init=domain_embeddings.T, size_min=len(texts) // 8)
        clusterer['kmeans'] = kmeans
        out = clusterer['kmeans'].fit_predict(np.concatenate([texts[domain]['vecs'] for domain in texts], 0))
        args.output_clusters.mkdir(exist_ok=True)
        with open(args.output_clusters / "clusters.pkl", "wb+") as f:
            pickle.dump(clusterer, f)

        if args.pretrain_clusters_only:
            sys.exit(1)
    elif args.load:
        with open(args.load / "clusters.pkl", "rb") as f:
            clusterer = pickle.load(f)
    else:
        clusterer=None

    if args.train_files:
        with open(args.train_files, 'r') as f:
            args_train_files = [x.strip() for x in f.readlines()]
    else:
        args_train_files = None

    if args.dev_files:
        with open(args.dev_files, 'r') as f:
            args_dev_files = [x.strip() for x in f.readlines()]
    else:
        args_dev_files = None

    if args.test_files:
        with open(args.test_files, 'r') as f:
            args_test_files = [x.strip() for x in f.readlines()]
    else:
        args_test_files = None

    if not args.from_file:
        resolved_path = str(PROJECT_DIR / domain / domain)
        # if args.domain in ['reddit', '1b', 'openwebtext']:
        #     domain_files = None
        # else:
        with open(PROJECT_DIR / args.domain  / "metadata" / "filenames.txt", 'r') as f:
            domain_files = []
            for x in tqdm(f.readlines()):
                fp = x.strip()
                domain_files.append(fp)
    else:
        domain_files = None

    if args.domain in ['reddit', '1b']:
        add_bos_token = False
        num_workers = args.num_workers
        batch_size = args.batch_size
    else:
        add_bos_token = True
        num_workers = args.num_workers
        batch_size = args.batch_size
    if not args.test_only and not args.dev_only:
        train_files, train_files_to_ignore, num_train_tokens = write_split(args.domain,
                                                        add_bos_token,
                                                        num_workers,
                                                        batch_size,
                                                        output_dir,
                                                        "train",
                                                        files=args_train_files or domain_files,
                                                        clusterer=clusterer,
                                                        anonymize=args.anonymize)
    else:
        train_files = None
        train_files_to_ignore = None
        num_train_tokens = None

    if not args.train_only:
        if not args.test_only:
            if args.from_file:
                train_files_to_ignore = None
            dev_files, dev_files_to_ignore, num_dev_tokens = write_split(args.domain,
                                                        add_bos_token,
                                                        num_workers,
                                                        batch_size,
                                                        output_dir,
                                                        "dev",
                                                        files=args_dev_files or domain_files,
                                                        ignore_files=args_train_files or train_files_to_ignore,
                                                        clusterer=clusterer,
                                                        from_file=args.from_file)
        else:
            dev_files = None
            dev_files_to_ignore = None
            num_dev_tokens = None
        if not args.dev_only:
            if args.from_file:
                train_files_to_ignore = []
                dev_files_to_ignore = []
            if args_train_files and args_dev_files:
                ignore_files = args_train_files + args_dev_files
            else:
                ignore_files = train_files_to_ignore + dev_files_to_ignore

            test_files, test_files_to_ignore, num_test_tokens = write_split(args.domain,
                                                add_bos_token,
                                                num_workers,
                                                batch_size,
                                                output_dir,
                                                "test",
                                                files=args_test_files or domain_files,
                                                ignore_files=ignore_files,
                                                clusterer=clusterer,
                                                from_file=args.from_file)

        else:
            test_files = None
            test_files_to_ignore = None
            num_test_tokens = None
    if train_files_to_ignore:
        with open(args.output_dir / "train_files.txt", "w+") as f:
            for file in train_files_to_ignore:
                f.write(str(file) + "\n")
    if dev_files_to_ignore:
        with open(args.output_dir / "dev_files.txt", "w+") as f:
            for file in dev_files_to_ignore:
                f.write(str(file) + "\n")
    if test_files_to_ignore:
        with open(args.output_dir / "test_files.txt", "w+") as f:
            for file in test_files_to_ignore:
                f.write(str(file) + "\n")


    print("Finished successfully.")
    if num_train_tokens:
        with open(args.output_dir / "train_token_counts.txt", 'w+') as f:
            json.dump(num_train_tokens, f)
    if num_dev_tokens:
        with open(args.output_dir / "dev_token_counts.txt", 'w+') as f:
            json.dump(num_dev_tokens, f)
    if num_test_tokens:
        with open(args.output_dir / "test_token_counts.txt", 'w+') as f:
            json.dump(num_test_tokens, f)
