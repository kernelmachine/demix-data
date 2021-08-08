import os
from typing import Optional, List, Tuple, Iterable, Any
from collections import defaultdict
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import gzip
from joblib import Parallel, delayed

from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS
from domain_loader.utils import take_n_tokens, REGEXES
from tqdm.auto import tqdm
import numpy as np
from scipy import sparse
import re


def reservoir_sampling(iterator: Iterable[Any], K: int):
    """
    Sample from an iterator without loading the iterator into memory.
    """
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item
    return result


class Domain(Dataset):
    def __init__(self,
                 domain_directory: Path,
                 filenames: Optional[List[str]] = None,
                 add_bos_token: bool = False,
                 bos_token: str = "<|endoftext|>",
                 ignore_files: Optional[List[str]] = [],
                 sample: int = None,
                 sample_from_head: bool = False,
                 track_token_count: bool = False,
                 anonymize: bool = False,
                 sample_by_metadata: Optional[Tuple[str, int]] = None,
                 metadata_columns: List[str] = None,
                 metadata_file: Optional[Path] = None,
                 **metadata_filters):
        """
        Basic domain dataset.

        Arguments
        =========
        domain_directory -> root directory of the domain
        filenames -> list of filenames to use (avoids scanning the directory, which may take a while)
        add_bos_token -> prepend a beginning of sentence token to each document during loading
        ignore_files -> list of filenames to ignore (use to specify, for example, dev or test data you'd like to ignore)
        sample -> specify number of random documents from the domain to sample
        sample_from_head -> if set, will sample from the head of the domain, rather than doing reservoir sampling, which may take a while, though it is more unbiased.
        track_token_count -> if set, will track the number of tokens sampled during data loading.
        anonymize -> if set, will apply some basic regexes to loaded data to redact user identifiable information.
        sample_by_metadata -> if set, in the form (metadata_column, k), sample k documents that align with the metadata_column.
        metadata_columns -> if set, return metadata columns (from metadata_file) for each document
        metadata_file -> if set, read metadata_file as well
        **metadata_filters -> if set, in the form {metadata_column: [item_1,item_2,...]}, will filter documents that satisfy these metadata filters
        """
        super().__init__()
        self.add_bos_token = add_bos_token
        self.anonymize = anonymize

        self.anonymizer = {re.compile(regex['regex']): regex['repl'] for regex in REGEXES}

        self.bos_token = bos_token
        self.domain_directory = domain_directory
        self.files = {}

        if metadata_file:
            print(f'loading files from metadata in {metadata_file}')
            with open(metadata_file, 'r') as f:
                for ix, line in tqdm(enumerate(f)):
                    if sample:
                        if ix > sample:
                            break
                    z = json.loads(line)
                    if filenames:
                        if z['filename'] not in filenames:
                            continue
                    metadata_ = {metadata_column: z[metadata_column] for metadata_column in metadata_columns}
                    if metadata_filters:
                        for key, items in metadata_filters.items():
                            if metadata_[key] in items:
                                self.files[z['filename']] = metadata_
                    else:
                        self.files[z['filename']] = metadata_

            if sample_by_metadata:
                files_ = {}
                self.metadata_counts = defaultdict(int)
                for file in self.files:
                    if self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] < sample_by_metadata['sample_size']:
                        self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] += 1
                        files_[file] = self.files[file]
                self.files = files_
        else:
            self.metadata_file = None
            if filenames:
                if isinstance(filenames[0], Tuple):
                    self.files = dict(filenames)
                else:
                    self.files = filenames
            else:
                fs = tqdm(domain_directory.glob("*/*"))
                if sample:
                    print(f"Loading {sample} files from {domain_directory}...")
                    if sample_from_head:
                        sample_files = []
                        for ix, file in enumerate(fs):
                            if ix < sample:
                                sample_files.append(file)
                            else:
                                break
                    else:
                        sample_files = reservoir_sampling(fs, sample)
                    self.files = sample_files
                else:
                    print(f"Loading all files from {domain_directory}...")
                    self.files = list(fs)

        if ignore_files:
            self.files = list(set(self.files) - set(ignore_files))
        print(f"loaded {len(self.files)} files, ignoring {len(ignore_files)} files")

    def __getitem__(self, idx):
        if self.metadata_file:
            x = self.files[idx]
            file, metadata = str(x[0]), x[1]
        else:
            file = str(self.files[idx])
            metadata = []
        try:
            if file.endswith('.gz'):
                with gzip.open(file, 'rb') as f:
                    text = f.read().decode('utf-8')
            else:
                with open(file, "r") as f:
                    text = f.read()
        except:
            text = ""
        if self.add_bos_token:
            text = self.bos_token + " " + text
        if self.anonymize:
            for x,y in self.anonymizer.items():
                text = x.sub(y, text)
        token_count = len(text.split())
        return file, text, token_count, metadata

    def __len__(self):
        return len(self.files)



class DomainTokenized(Domain):
    def __init__(self,
                 domain_directory: Path,
                 metadata_file: Optional[Path] = None,
                 filenames: Optional[List[str]] = None,
                 tokenizer: Optional[GPT2Tokenizer] = None,
                 ignore_files: Optional[List[str]] = [],
                 **metadata_filters):
        """
        Domain dataset with tokenization built in.
        """
        super().__init__(domain_directory=domain_directory,
                        metadata_file=metadata_file,
                        filenames=filenames,
                        ignore_files=ignore_files,
                        **metadata_filters)
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer

    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text, token_count, metadata = super().__getitem__(idx)
        tokenized_text = np.array(self.tokenizer.encode(text, truncation=True))
        return filename, tokenized_text, token_count, metadata


class DomainVectorized(Domain):
    def __init__(self,
                 domain_directory: Path,
                 vectorizer = None,
                 filenames: Optional[List[str]] = None,
                 add_bos_token: bool = False,
                 metadata_columns: List[str] = None,
                 ignore_files: Optional[List[str]] = [],
                 sample_by_metadata: Optional[Tuple[str, int]] = None,
                 metadata_file: Optional[Path] = None,
                 tokenizer = None,
                 sample: int = None,
                 **metadata_filters):
        """
        Domain dataset with document vectorization built in.
        """
        super().__init__(domain_directory=domain_directory,
                 filenames=filenames,
                 add_bos_token=add_bos_token,
                 metadata_columns=metadata_columns,
                 ignore_files=ignore_files,
                 sample_by_metadata=sample_by_metadata,
                 metadata_file=metadata_file,
                 sample=sample,
                 **metadata_filters)
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer
        if not vectorizer:
            self.vectorizer = TfidfVectorizer(vocabulary=self.tokenizer.encoder, max_features=len(self.tokenizer.encoder))
        else:
            self.vectorizer = vectorizer
        print("fitting vectorizer...")

    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text, token_count, metadata = super().__getitem__(idx)
        vectorized_text = self.vectorizer(tokenized_text)
        return filename, vectorized_text, token_count, metadata


def domain_dataloader(domain_directory: Path,
                      metadata_file: Optional[Path] = None,
                      filenames: Optional[List[str]] = None,
                      num_workers: int = 16,
                      batch_size: int = 16,
                      **metadata_filters):

    if tokenized:
        dataset = DomainTokenized(domain_directory, metadata_file, filenames, **metadata_filters)
    else:
        dataset = Domain(domain_directory, metadata_file, filenames, **metadata_filters)
    dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size)
    return dataloader
