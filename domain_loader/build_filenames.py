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

from domain_loader.constants import PROJECT_DIR
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from domain_loader.domain_loader import Domain

if __name__ == '__main__':

    domains = ["1b", "realnews", "med", "legal", "openwebtext", "reddit", "cs", "reviews"]
    
    for domain in domains:
        dataset = Domain(PROJECT_DIR / domain / domain)
        
        if domains in ["1b", "reddit"]:
            num_workers = 1
            batch_size = 1
        else:
            num_workers = 16
            batch_size=1024
        dataloader = DataLoader(dataset,
                                num_workers=16,
                                batch_size=16)
        
        pbar = tqdm(dataloader)
        curr_files = 0
        (PROJECT_DIR / domain / "metadata" ).mkdir(exist_ok=True)
        with open(PROJECT_DIR / domain / "metadata" / "filenames.txt", "w+") as f:
            for fname, _, _, _ in pbar:
                for fn in fname:
                    f.write(fn + "\n")
                    curr_files += 1
        print(f"Number of files in {str(PROJECT_DIR / domain / domain)}: {curr_files}")
