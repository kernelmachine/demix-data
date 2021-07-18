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

from domain_loader.constants import PROJECT_DIR
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from domain_loader.domain_loader import Domain
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain')
    args = parser.parse_args()
    domain = args.domain
   
    dataset = Domain(PROJECT_DIR / domain/domain)
    
    dataloader = DataLoader(dataset,
                            num_workers=16,
                            batch_size=16)
    
    pbar = tqdm(dataloader)
    curr_tokens = 0
    for _, _, token_count, _ in pbar:
        curr_tokens += sum(token_count)
        pbar.set_description(f"{humanize.intword(curr_tokens)} tokens")

    print(f"Number of tokens in {str(PROJECT_DIR / domain / domain)}: {humanize.intword(curr_tokens)}")
