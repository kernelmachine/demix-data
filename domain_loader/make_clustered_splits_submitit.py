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
import time

import humanize
import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import TypeVar, Iterable, List, Sequence, Union, Any
import submitit


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
                                            max_length=2048,
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
        vec = featurizer(text).cpu().numpy()
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

    job_env = submitit.JobEnvironment()

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
        model = AutoModel.from_pretrained(path_to_featurizer).cuda(job_env.global_rank)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--path-to-clusterer", type=str)
    parser.add_argument("--path-to-featurizer", type=str)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--log-dir", type=str, default='logs')

    args = parser.parse_args()
    domain = args.domain

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

    
    clusterer = load_model(args.path_to_clusterer)
    path_to_featurizer = args.path_to_featurizer

    # if args.train_files:
    #     with open(args.train_files, 'r') as f:
    #         args_train_files = [x.strip() for x in f.readlines()]
    # else:
    #     args_train_files = None

    # if args.dev_files:
    #     with open(args.dev_files, 'r') as f:
    #         args_dev_files = [x.strip() for x in f.readlines()]
    # else:
    #     args_dev_files = None

    # if args.test_files:
    #     with open(args.test_files, 'r') as f:
    #         args_test_files = [x.strip() for x in f.readlines()]
    # else:
    #     args_test_files = None

    # if not args.from_file and not args.use_iterable_dataset:
    #     resolved_path = str(DATA_DIR / domain / domain)
    #     fnames_file = DATA_DIR / args.domain  / "metadata" / "filenames.txt"
    #     with open(fnames_file, 'r') as f:
    #         domain_files = []
    #         for x in tqdm(f.readlines()):
    #             fp = x.strip()
    #             domain_files.append(fp)
    # else:
    #     domain_files = None

    # if args.domain in ['reddit', '1b']:
    #     add_bos_token = False
    #     num_workers = args.num_workers
    #     batch_size = args.batch_size
    # else:
    #     add_bos_token = True
    #     num_workers = args.num_workers
    #     batch_size = args.batch_size
    # if not args.test_only and not args.dev_only:
    #     train_ids, num_train_tokens = write_split(args.domain,
    #                                                     output_dir,
    #                                                     "train",
    #                                                     add_bos_token,
    #                                                     num_workers=num_workers,
    #                                                     batch_size=batch_size,
    #                                                     files=args_train_files or domain_files,
    #                                                     from_file=args.from_file,
    #                                                     clusterer=clusterer,
    #                                                     path_to_featurizer=path_to_featurizer,
    #                                                     anonymize=args.anonymize,
    #                                                     text_field=args.text_field,
    #                                                     use_iterable_dataset=args.use_iterable_dataset)
    # else:
    #     train_ids = None
    #     num_train_tokens = None
    # if not args.train_only:
    #     if not args.test_only:
    #         if args.from_file:
    #             train_files_to_ignore = None
    #         dev_ids, num_dev_tokens = write_split(args.domain,
    #                                                 output_dir,
    #                                                 "dev",
    #                                                 add_bos_token,
    #                                                 num_workers=num_workers,
    #                                                 batch_size=batch_size,
    #                                                 files=args_dev_files or domain_files,
    #                                                 ignore_ids=train_ids,
    #                                                 clusterer=clusterer,
    #                                                 path_to_featurizer=path_to_featurizer,
    #                                                 from_file=args.from_file,
    #                                                 text_field=args.text_field,
    #                                                 use_iterable_dataset=args.use_iterable_dataset)
    #     else:
    #         dev_ids = None
    #         num_dev_tokens = None
    #     if not args.dev_only:
    #         test_ids, num_test_tokens = write_split(
    #                                             args.domain,
    #                                             output_dir,
    #                                             "test",
    #                                             add_bos_token,
    #                                             num_workers=num_workers,
    #                                             batch_size=batch_size,
    #                                             files=args_test_files or domain_files,
    #                                             ignore_ids={**train_ids, **dev_ids},
    #                                             clusterer=clusterer,
    #                                             path_to_featurizer=path_to_featurizer,
    #                                             from_file=args.from_file,
    #                                             text_field=args.text_field,
    #                                             use_iterable_dataset=args.use_iterable_dataset)
    #     else:
    #         test_files = None
    #         test_files_to_ignore = None
    #         num_test_tokens = None
    #         test_ids = None
    # else:
    #     dev_ids = None
    #     dev_files_to_ignore = None
    #     num_dev_tokens = None
    #     test_files = None
    #     dev_files = None
    #     test_files_to_ignore = None
    #     num_test_tokens = None
    #     test_ids = None

    # if args.submitit:
        # files = np.array_split(df, args.num_shards)


        # train_ids, num_train_tokens = write_split(args.domain,
        #                                                 output_dir,
        #                                                 "train",
        #                                                 add_bos_token,
        #                                                 num_workers=num_workers,
        #                                                 batch_size=batch_size,
        #                                                 files=args_train_files or domain_files,
        #                                                 from_file=fil,
        #                                                 clusterer=clusterer,
        #                                                 path_to_featurizer=path_to_featurizer,
        #                                                 anonymize=args.anonymize,
        #                                                 text_field=args.text_field,
        #                                                 use_iterable_dataset=args.use_iterable_dataset)
    def splitter(input_file, output_dir, split, domain, clusterer, path_to_featurizer):
        _ = write_split(domain,
                        output_dir,
                        split,
                        False,
                        num_workers=1,
                        batch_size=8,
                        files=None,
                        from_file=input_file,
                        clusterer=clusterer,
                        path_to_featurizer=path_to_featurizer,
                        anonymize=False,
                        text_field="text",
                        use_iterable_dataset=True)
        return
    input_files = [args.input_dir / x for x in os.listdir(str(args.input_dir))]
    output_dirs = [ args.output_dir  / str(ix)  for ix in range(len(input_files))]    

    log_folder = f"{args.log_dir}/%j"

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(slurm_array_parallelism=len(input_files),
                            timeout_min=400,
                            slurm_partition="learnlab,devlab",
                            cpus_per_task=10,
                            tasks_per_node=1,
                            gpus_per_node=1 if "tfidf" not in path_to_featurizer else 0)
    jobs = executor.map_array(splitter,
                            input_files,
                            output_dirs,
                            ["train"] * len(input_files),
                            [args.domain] * len(input_files),
                            [clusterer] * len(input_files),
                            [path_to_featurizer] * len(input_files))
    print('submitted job!')
    num_finished = sum(job.done() for job in jobs)
    pbar = tqdm(total=len(jobs))
    pbar.set_description('job completion status')

    while num_finished < len(jobs):
        # wait and check how many have finished
        time.sleep(5)   
        curr_finished = sum(job.done() for job in jobs)
        if curr_finished != num_finished:
            num_finished = curr_finished
            pbar.update(1)
    pbar.close()
