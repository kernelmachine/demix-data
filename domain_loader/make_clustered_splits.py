import argparse
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
from fairseq.data.encoders.gpt2_bpe import get_encoder

import humanize
import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import TypeVar, Iterable, List, Sequence, Union, Any


from domain_loader.constants import DATA_DIR, TOKEN_COUNTS,TOKEN_COUNTS_DEMIX
from domain_loader.domain_loader import Domain, IterableDomain
from domain_loader.utils import take_n_tokens
from cluster.cluster import  load_model
from transformers import AutoModel, AutoTokenizer
T = TypeVar('T')


def hf_extract_features(lines, model, tokenizer):
    input_ids = tokenizer.batch_encode_plus(lines,
                                            add_special_tokens=True,
                                            truncation=True,
                                            max_length=model.max_length,
                                            padding='max_length',
                                            return_tensors='pt')
    last_non_masked_idx = torch.sum(input_ids['attention_mask'], dim=1) - 1
    # last_non_masked_idx = last_non_masked_idx.view(-1, 1).repeat(1, 768).unsqueeze(1).cuda()
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        vectors_ = model(**input_ids)
        #vs = []
        # for ix, idx in enumerate(last_non_masked_idx):
            # vs.append(out[0][ix, :idx, :])
        return vectors_[0].sum(axis=1) / input_ids['attention_mask'].sum(axis=-1).unsqueeze(-1)

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

def get_cluster_id(clusterer, featurizer, featurizer_type, text):
    # text = [x.replace("<|endoftext|>", "") for x in text]
    if featurizer_type == 'tfidf':
        vec = featurizer.transform(text)
    else:
        vec = featurizer(text)
    # vec = clusterer['svd'].transform(vec)
    cluster_id = clusterer['model'].predict(vec)
    return cluster_id

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def read_chunks(fileObj, chunkSize=50000, batch_size=2048):
    """
    Lazy function to read a file piece by piece.
    Default chunk size: 2kB.

    """
    while True:
        data = fileObj.read(chunkSize)
        if not data:
            break
        v = data.split()
        for chunk in chunks(v, batch_size):
            yield " ".join(chunk)
            
def write_split(domain: str,
                output_dir: str,
                split: str,
                add_bos_token: bool,
                bos_token: str = "<|endoftext|>",
                num_workers: int = 16,
                batch_size: int = 16,
                files=None,
                ignore_files=[],
                ignore_ids={},
                clusterer=None,
                path_to_featurizer=None,
                from_file=None,
                text_field="text",
                use_iterable_dataset=False,
                anonymize=False):

    if not from_file:
        
        if use_iterable_dataset:
            dataset = IterableDomain(DATA_DIR / domain / domain,
                         add_bos_token=add_bos_token,
                         bos_token=bos_token,
                         text_field=text_field,
                         ignore_ids=ignore_ids,
                         anonymize=anonymize)
        else:
            dataset = Domain(DATA_DIR / domain / domain,
                            filenames=files,
                            add_bos_token=add_bos_token,
                            bos_token=bos_token,
                            ignore_files=ignore_files,
                            anonymize=anonymize)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    else:
        fh = open(from_file, 'r')
        loader = batchify(read_chunks(fh), batch_size)
    files = []
    done = False

    if clusterer:
        print("Detected clusterer. Clustering while loading splits.")
        dirs = [output_dir / str(i) for i in range(clusterer['model'].n_clusters)]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)
        filehandle = [open(output_dir / str(i) / f"{split}.txt", "w+") for i in range(clusterer['model'].n_clusters)]
        curr_tokens = dict((i, 0) for i in range(clusterer['model'].n_clusters))
    else:
        filehandle = [open(output_dir / f"{split}.txt", "w+")]
        curr_tokens = {0: 0}
    
    if 'tfidf' in path_to_featurizer:
        featurizer = load_model(path_to_featurizer)
        featurizer_type = 'tfidf'
    else:
        model = AutoModel.from_pretrained(path_to_featurizer)
        tokenizer = AutoTokenizer.from_pretrained(path_to_featurizer)
        featurizer = lambda x: hf_extract_features(x, model, tokenizer)
        featurizer_type = 'hf'
    pbar = tqdm(loader)
    written = False
    ids = {}
    continuer=False
    complete = {}
    if not from_file:
        for id, fname, text,_, _ in pbar:
            if ignore_ids:
                for x,y in zip(fname, id):
                    if ignore_ids.get(f"{x}_{y}"):
                        continuer = True
                        continue
            if continuer:
                continuer = False
                continue
            for x,y in zip(fname, id):
                ids[f"{x}_{y}"] = 1
            if clusterer:
                if domain in ['1b', 'reddit']:
                    tt = [t.split("<|endoftext|>") for t in text]
                    text = ["<|endoftext|> " + y for x in tt for y in x]
                cluster_ids = get_cluster_id(clusterer, featurizer, featurizer_type, text)
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
                    if use_iterable_dataset:
                        if not complete.get(fname[-1]):
                            complete[fname[-1]] = 1
                        s = f"{split}, total shards: {dataset.num_files}, progress: {round(len(complete) / dataset.num_files * 100)}%, num tokens: {humanize.intword(curr_tokens[0])}"
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
        bpe = get_encoder("/private/home/suching/raw_data/demix_scale/data-bin/encoder.json", "/private/home/suching/raw_data/demix_scale/data-bin/vocab.bpe")

        for bpe_tokens in pbar:
            if continuer:
                continuer = False
                continue
            text = [bpe.decode([int(x) for x in x.split()]) for x in bpe_tokens]
            
            if clusterer:
                if domain in ['1b', 'reddit']:
                    tt = [t.split("<|endoftext|>") for t in text]
                    text = ["<|endoftext|> " + y for x in tt for y in x]
                cluster_ids = get_cluster_id(clusterer, featurizer, featurizer_type, text)
                iter_ = zip(cluster_ids, text)
            else:
                iter_ = text
            for item, bpe_repr in zip(iter_, bpe_tokens):
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
                    if use_iterable_dataset:
                        if not complete.get(fname[-1]):
                            complete[fname[-1]] = 1
                        s = f"{split}, total shards: {dataset.num_files}, progress: {round(len(complete) / dataset.num_files * 100)}%, num tokens: {humanize.intword(curr_tokens[0])}"
                    else:
                        s = f"{split}, num tokens: {humanize.intword(curr_tokens[0])}"
                pbar.set_description(s)
                if sum(curr_tokens.values()) > TOKEN_COUNTS_DEMIX[domain][f'num_{split}_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(doc.split()[:TOKEN_COUNTS_DEMIX[domain][f'num_{split}_tokens']])
                        if clusterer:
                            filehandle[cluster_id].write(bpe_repr.strip() + "\n")
                            curr_tokens[cluster_id] += len(doc.split())
                        else:
                            filehandle[0].write(doc.strip() + "\n")
                            curr_tokens[0] += len(doc.split())
                        written = True
                    done = True
                    break
                if clusterer:
                    filehandle[cluster_id].write(bpe_repr.strip() + "\n")
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
        return None, curr_tokens
    else:
        return ids, curr_tokens

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
    parser.add_argument("--path-to-clusterer", type=str)
    parser.add_argument("--path-to-featurizer", type=str)

    parser.add_argument("--from-file", type=Path)
    parser.add_argument("--train-only", action='store_true')
    parser.add_argument("--dev-only", action='store_true')
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--num-train-tokens", type=int, default=None)
    parser.add_argument("--num-dev-tokens", type=int, default=None)
    parser.add_argument("--num-test-tokens", type=int, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--anonymize", action='store_true')
    parser.add_argument("--use-iterable-dataset", action='store_true')

    args = parser.parse_args()
    domain = args.domain

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

    if args.num_train_tokens:
        TOKEN_COUNTS[domain]['num_train_tokens'] = args.num_train_tokens
    if args.num_dev_tokens:
        TOKEN_COUNTS[domain]['num_dev_tokens'] = args.num_dev_tokens
    if args.num_test_tokens:
        TOKEN_COUNTS[domain]['num_test_tokens'] = args.num_test_tokens
    
    clusterer = load_model(args.path_to_clusterer)
    path_to_featurizer = args.path_to_featurizer

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

    if not args.from_file and not args.use_iterable_dataset:
        resolved_path = str(DATA_DIR / domain / domain)
        fnames_file = DATA_DIR / args.domain  / "metadata" / "filenames.txt"
        with open(fnames_file, 'r') as f:
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
        train_ids, num_train_tokens = write_split(args.domain,
                                                        output_dir,
                                                        "train",
                                                        add_bos_token,
                                                        num_workers=num_workers,
                                                        batch_size=batch_size,
                                                        files=args_train_files or domain_files,
                                                        from_file=args.from_file,
                                                        clusterer=clusterer,
                                                        path_to_featurizer=path_to_featurizer,
                                                        anonymize=args.anonymize,
                                                        text_field=args.text_field,
                                                        use_iterable_dataset=args.use_iterable_dataset)
    else:
        train_ids = None
        num_train_tokens = None
    if not args.train_only:
        if not args.test_only:
            if args.from_file:
                train_files_to_ignore = None
            dev_ids, num_dev_tokens = write_split(args.domain,
                                                    output_dir,
                                                    "dev",
                                                    add_bos_token,
                                                    num_workers=num_workers,
                                                    batch_size=batch_size,
                                                    files=args_dev_files or domain_files,
                                                    ignore_ids=train_ids,
                                                    clusterer=clusterer,
                                                    path_to_featurizer=path_to_featurizer,
                                                    from_file=args.from_file,
                                                    text_field=args.text_field,
                                                    use_iterable_dataset=args.use_iterable_dataset)
        else:
            dev_ids = None
            num_dev_tokens = None
        if not args.dev_only:


            test_ids, num_test_tokens = write_split(
                                                args.domain,
                                                output_dir,
                                                "test",
                                                add_bos_token,
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                files=args_test_files or domain_files,
                                                ignore_ids={**train_ids, **dev_ids},
                                                clusterer=clusterer,
                                                path_to_featurizer=path_to_featurizer,
                                                from_file=args.from_file,
                                                text_field=args.text_field,
                                                use_iterable_dataset=args.use_iterable_dataset)
        else:
            test_files = None
            test_files_to_ignore = None
            num_test_tokens = None
            test_ids = None
    else:
        dev_ids = None
        dev_files_to_ignore = None
        num_dev_tokens = None
        test_files = None
        dev_files = None
        test_files_to_ignore = None
        num_test_tokens = None
        test_ids = None

    if train_ids:
        with open(args.output_dir / "train_ids.txt", "w+") as f:
            for file in train_ids:
                f.write(str(file) + "\n")
    if dev_ids:
        with open(args.output_dir / "dev_ids.txt", "w+") as f:
            for file in dev_ids:
                f.write(str(file) + "\n")
    if test_ids:
        with open(args.output_dir / "test_ids.txt", "w+") as f:
            for file in test_ids:
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

    if num_train_tokens:
        print(f"Num train tokens: {humanize.intword(sum(num_train_tokens.values()))}")
    if num_dev_tokens:
        print(f"Num dev tokens: {humanize.intword(sum(num_dev_tokens.values()))}")
    if num_test_tokens:
        print(f"Num test tokens: {humanize.intword(sum(num_test_tokens.values()))}")
