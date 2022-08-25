import os
from typing import Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import humanize
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import GPT2Tokenizer
import pandas as pd
from pathlib import Path
import gzip

from domain_loader.constants import DATA_DIR
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from domain_loader.domain_loader import Domain, IterableDomain
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--path-to-filenames', type=str, default=None)
    parser.add_argument('--text-field', type=str, default='text')
    
    parser.add_argument('--use-iterable-dataset', action='store_true')


    args = parser.parse_args()
    domain = args.domain

    if args.path_to_filenames:
        with open(args.path_to_filenames, 'r') as f:
            filenames = [x.strip() for x in f.readlines()]
    else:
        filenames = None
    if args.use_iterable_dataset:
        dataset = IterableDomain(DATA_DIR / args.domain / args.domain, text_field=args.text_field)
    else:
        dataset = Domain(DATA_DIR / args.domain , filenames=filenames, text_field=args.text_field)

    dataloader = DataLoader(dataset,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size)

    pbar = tqdm(dataloader)
    curr_tokens = 0
    complete = {}
    for id, file,text, token_count, _ in pbar:
        curr_tokens += sum(token_count)
        if not complete.get(file[-1]):
            complete[file[-1]] = 1 
        pbar.set_description(f"total shards: {dataset.num_files}, progress: {round(len(complete) / dataset.num_files * 100) }%, {humanize.intword(curr_tokens)} tokens")

    print(f"Number of tokens in {str(DATA_DIR / domain / domain)}, {args.path_to_filenames}: {humanize.intword(curr_tokens)}")
