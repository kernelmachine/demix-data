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

    domain = "gutenberg"
   
    dataset = Domain(PROJECT_DIR / domain)
    
    dataloader = DataLoader(dataset,
                            num_workers=16,
                            batch_size=16)
    
    pbar = tqdm(dataloader)
    curr_tokens = 0
    for fname, text, _ in pbar:
        for item in text:
            curr_tokens += len(item.split())

    print(f"Number of tokens in {str(PROJECT_DIR / domain / domain)}: {curr_tokens}")
