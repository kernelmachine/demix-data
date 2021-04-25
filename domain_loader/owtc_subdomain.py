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

def write_split(domains, subdomain, add_bos_token, num_workers, batch_size, output_dir, split, files=None, ignore_files=[]):
    metadata_df = pd.read_json(PROJECT_DIR / args.domain / "metadata" / "metadata.1.jsonl", lines=True)
    populated_domains = list(metadata_df.domain.value_counts().head(n=1000).index)
    pbar = tqdm(populated_domains)


    for subdomain in pbar:
        files = list(metadata_df.loc[metadata_df.domain == subdomain].filename)
        dataset = Domain(PROJECT_DIR / domain / domain,
                        filenames=files,
                        add_bos_token=add_bos_token,
                        ignore_files=ignore_files,
                        domain=subdomain)
        
        loader = DataLoader(dataset, num_workers=1, batch_size=1)
        
        files = []
        done = False
        (output_dir / subdomain).mkdir(exist_ok=True)
        curr_tokens = 0
        with open(output_dir / subdomain /  f"{split}.txt", "w+") as f:
            pbar = loader
            
            written = False
            for fname, text, _ in pbar:
                files.extend(fname)
                for item in text:
                    curr_tokens += len(item.split())
                    # pbar.set_description(f"{split}, num tokens: {humanize.intword(curr_tokens)}")
                    if curr_tokens > 10000:
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
    parser.add_argument("--subdomain")

    parser.add_argument("--add-bos-token", action='store_true')
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    domain = args.domain
    
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    train_files, train_files_to_ignore, num_train_tokens = write_split(args.domain,
                                                    args.subdomain,
                                                     args.add_bos_token,
                                                     args.num_workers,
                                                     args.batch_size,
                                                     output_dir,
                                                     "train")


    print("Finished successfully.")
    print(f"Num train tokens: {humanize.intword(num_train_tokens)}")

