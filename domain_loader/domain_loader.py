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

from loader.constants import PROJECT_DIR, TOKEN_COUNTS
from loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np


class Domain(Dataset):
    def __init__(self,
                 domain_directory: Path,
                 filenames: Optional[List[str]] = None,
                 add_bos_token: bool = False,
                 ignore_files: Optional[List[str]] = [],
                 metadata_directory: Optional[Path] = None,
                 **metadata_filters):
        super().__init__()      
        self.add_bos_token = add_bos_token 
        self.bos_token = "<|endoftext|> " 
        self.domain_directory = domain_directory
        if metadata_filters:
            if metadata_directory is None:
                raise Exception("metadata_directory cannot be none if metadata_filters applied.")
            print("metadata filter detected, loading metadata...")
            metadata = pd.read_json(metadata_directory / "metadata.jsonl", lines=True)
            
            fnames = []
            for key, items in metadata_filters.items():
                m = metadata.loc[metadata[key].isin(items)]
                if 'filenames' in m.columns:
                    m.filenames.apply(lambda x: fnames.extend(x))
                else:
                    fnames.extend(m.filename.values)
            filenames = list(set(fnames))
            filenames = [domain_directory / Path(filename) for filename in filenames]

        if filenames is not None:
            print(f"Loading list of files from {domain_directory}...")
            self.files = filenames
            assert all(file.exists for file in self.files)
        else:
            print(f"Loading all files from {domain_directory}...")
            self.files = list(tqdm(domain_directory.glob("*/*")))
        
        if ignore_files:
            self.files = set([str(x) for x in self.files]) - set(ignore_files)
            self.files = [Path(x) for x in self.files]
        print(f"loaded {len(self.files)} files, ignoring {len(ignore_files)} files")

    def __getitem__(self, idx):
        file = self.files[idx].resolve()
        if file.name.endswith('.gz'):
            with gzip.open(file, 'rb') as f:
                text = f.read().decode('utf-8')
        else:
            text = file.read_text(errors='ignore')
        if self.add_bos_token:
            text = self.bos_token + text
        return str(file), text

    def __len__(self):
        return len(self.files)


class DomainTokenized(Domain):
    def __init__(self,
                 domain_directory: Path,
                 metadata_directory: Optional[Path] = None,
                 filenames: Optional[List[str]] = None,
                 tokenizer: Optional[GPT2Tokenizer] = None,
                 ignore_files: Optional[List[str]] = [],
                 **metadata_filters):
        super().__init__(domain_directory=domain_directory,
                        metadata_directory=metadata_directory,
                        filenames=filenames,
                        ignore_files=ignore_files,
                        **metadata_filters)
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer

    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text = super().__getitem__(idx)
        return filename, np.array(self.tokenizer.encode(text, truncation=True))


def domain_dataloader(domain_directory: Path,
                      metadata_directory: Optional[Path] = None,
                      filenames: Optional[List[str]] = None,
                      tokenized=False,
                      batch_size=64,
                      **metadata_filters):

    if tokenized:
        dataset = DomainTokenized(domain_directory, metadata_directory, filenames, **metadata_filters)
    else:
        dataset = Domain(domain_directory, metadata_directory, filenames, **metadata_filters)
    dataloader = DataLoader(dataset, num_workers=os.cpu_count(), batch_size=batch_size)
    return dataloader


