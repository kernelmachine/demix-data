import os
from typing import Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import GPT2Tokenizer
import pandas as pd
from pathlib import Path
import gzip

from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from domain_loader.domain_loader import Domain
import argparse
import humanize

def write_split(domain, add_bos_token, num_workers, batch_size, output_dir, split, files=None, ignore_files=[]):
    curr_tokens = 0
    dataset = Domain(PROJECT_DIR / domain / domain, filenames=files, add_bos_token=add_bos_token, ignore_files=ignore_files)
    
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    
    files = []
    done = False
    with open(output_dir / f"{split}.txt", "w+") as f:
        pbar = tqdm(loader)
        
        written = False
        for fname, text,_ in pbar:
            files.extend(fname)
            for item in text:
                curr_tokens += len(item.split())
                pbar.set_description(f"{split}, num tokens: {humanize.intword(curr_tokens)}")
                if curr_tokens > TOKEN_COUNTS[domain][f'num_{split}_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(item.split()[:TOKEN_COUNTS[domain][f'num_{split}_tokens']])
                        f.write(item + "\n")
                        written = True
                    done = True
                    break
                f.write(item + "\n")
                written = True
            if done:
                break
    return dataset.files, files, curr_tokens
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    parser.add_argument("--add-bos-token", action='store_true')
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    domain = args.domain
    
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    train_files, train_files_to_ignore, num_train_tokens = write_split(args.domain,
                                                     args.add_bos_token,
                                                     args.num_workers,
                                                     args.batch_size,
                                                     output_dir,
                                                     "train")
    dev_files, dev_files_to_ignore, num_dev_tokens = write_split(args.domain,
                                                 args.add_bos_token,
                                                 0,
                                                 args.batch_size,
                                                 output_dir,
                                                 "dev",
                                                 files=train_files,
                                                 ignore_files=train_files_to_ignore)
    _, _, num_test_tokens = write_split(args.domain,
                                        args.add_bos_token,
                                        0,
                                        args.batch_size,
                                        output_dir,
                                        "test",
                                        files=train_files,
                                        ignore_files=train_files_to_ignore + dev_files_to_ignore)


    print("Finished successfully.")
    print(f"Num train tokens: {humanize.intword(num_train_tokens)}")
    print(f"Num dev tokens: {humanize.intword(num_dev_tokens)}")
    print(f"Num test tokens: {humanize.intword(num_test_tokens)}")
