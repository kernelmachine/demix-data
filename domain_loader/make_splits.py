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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import TypeVar, Iterable, List, Sequence, Union, Any


from domain_loader.constants import DATA_DIR, TOKEN_COUNTS
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

def get_cluster_id(clusterer, text):
    text = [x.replace("<|endoftext|>", "") for x in text]
    vec = clusterer['vectorizer'].transform(text)
    vec = clusterer['svd'].transform(vec)
    cluster_id = clusterer['kmeans'].predict(vec)
    return cluster_id


def write_split(domain: str,
                output_dir: str,
                split: str,
                add_bos_token: bool,
                bos_token: str = "<|endoftext|>",
                num_workers: int = 16,
                batch_size: int = 16,
                files=None,
                ignore_files=[],
                clusterer=None,
                from_file=None,
                anonymize=False):

    if not from_file:
        dataset = Domain(DATA_DIR / domain / domain,
                         filenames=files,
                         add_bos_token=add_bos_token,
                         bos_token=bos_token,
                         ignore_files=ignore_files,
                         anonymize=anonymize)
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
        resolved_path = str(DATA_DIR / domain / domain)
        with open(DATA_DIR / args.domain  / "metadata" / "filenames.txt", 'r') as f:
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
                                                        output_dir,
                                                        "train",
                                                        add_bos_token,
                                                        num_workers=num_workers,
                                                        batch_size=batch_size,
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
            dev_files, dev_files_to_ignore, num_dev_tokens = write_split(
                                                        args.domain,
                                                        output_dir,
                                                        "dev",
                                                        add_bos_token,
                                                        num_workers=num_workers,
                                                        batch_size=batch_size,
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

            test_files, test_files_to_ignore, num_test_tokens = write_split(
                                                args.domain,
                                                output_dir,
                                                "test",
                                                add_bos_token,
                                                num_workers=num_workers,
                                                batch_size=batch_size,
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

    print(f"Num train tokens: {humanize.intword(sum(num_train_tokens.values()))}")
    print(f"Num dev tokens: {humanize.intword(sum(num_dev_tokens.values()))}")
    print(f"Num test tokens: {humanize.intword(sum(num_test_tokens.values()))}")
