import os
from typing import Optional, List, Tuple, Iterable, Any, Dict
from collections import defaultdict
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
import gzip
from joblib import Parallel, delayed
import itertools
from domain_loader.constants import DATA_DIR, TOKEN_COUNTS
from domain_loader.utils import take_n_tokens, REGEXES
from tqdm.auto import tqdm
import numpy as np
from scipy import sparse
import humanize
from fairseq.data import ConcatDataset
from itertools import chain

import re
import gzip

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

class IterableDomain(IterableDataset):
    def __init__(self,
                domain_directory: Path,
                add_bos_token: bool = False,
                bos_token: str = "<|endoftext|>",
                sample: int = None,
                sample_from_head: bool = False,
                track_token_count: bool = False,
                anonymize: bool = False,
                silent: bool=False,
                ignore_ids: Dict[str, int]={},
                text_field: str = "text",
                json: bool = True
                ):
        self.files = chain(domain_directory.glob(r"*.json.gz"), domain_directory.glob(r"*.jsonl.gz"), domain_directory.glob(r"*.txt.gz"))
        self.num_files = len(list(chain(domain_directory.glob(r"*.json.gz"), domain_directory.glob(r"*.jsonl.gz"), domain_directory.glob(r"*.txt.gz"))))
        self.add_bos_token = add_bos_token
        self.anonymize = anonymize
        self.anonymizer = {re.compile(regex['regex']): regex['repl'] for regex in REGEXES} 
        self.track_token_count = track_token_count
        self.bos_token = bos_token
        self.domain_directory = domain_directory
        self.text_field = text_field
        self.ignore_ids = ignore_ids
        self.json = json

    def line_mapper(self, line):
        if self.json:
            sample = json.loads(line)
            try:
                sample.get(self.text_field)
            except:
                sample = {self.text_field : line}
        else:
            sample = {self.text_field: line}
        if not sample.get(self.text_field):
            return None, None, None
        token_count = len(sample[self.text_field].split())
        text = sample.pop(self.text_field)
        if self.add_bos_token:
            text = self.bos_token + " " + text
        if self.anonymize:
            for x,y in self.anonymizer.items():
                text = x.sub(y, text)
        return text, token_count, sample
        
    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_total_num = 1
            worker_id = 0
        for json_file in itertools.islice(self.files, worker_id, None, worker_total_num):
            file_itr = gzip.open(json_file, 'rb')
            mapped_itr = map(self.line_mapper, file_itr)
            for ix, item in enumerate(mapped_itr):
                if item[0] is not None:
                    yield [ix, str(json_file)] + list(item)
                
    

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
                 silent: bool=False,
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
        self.metadata_file = metadata_file
        
        if filenames:
            if isinstance(filenames[0], Tuple):
                self.files = dict(filenames)
            else:
                self.files = filenames
        else:
            
            fs = tqdm(chain(domain_directory.glob("*"), domain_directory.glob("*/*"), domain_directory.glob("*/*/*")))
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
        if self.metadata_file:
            print(f'loading metadata in {metadata_file}')
            self.metadata_df = pd.read_json(self.metadata_file, lines=True)
            self.metadata_df['filename'] =  self.metadata_df.filename.apply(lambda x: str(domain_directory / x))
            self.metadata_df = self.metadata_df.loc[self.metadata_df.filename.isin(self.files)]
            self.metadata_columns = metadata_columns
            # self.files = 
            # metadata_df  = metadata_df.loc[metadata_df.filename ]
            # with open(metadata_file, 'r') as f:
            #     for ix, line in tqdm(enumerate(f)):
            #         z = json.loads(line)
            #         if filenames:
            #             if not self.files.get(domain_directory / z['filename']):
            #                 continue
                    
            #         metadata_ = {metadata_column: z[metadata_column] for metadata_column in metadata_columns}
            #         self.files[z['filename']] = metadata_
            # if sample_by_metadata:
            #     files_ = {}
            #     self.metadata_counts = defaultdict(int)
            #     for file in self.files:
            #         if self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] < sample_by_metadata['sample_size']:
            #             self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] += 1
            #             files_[file] = self.files[file]
            #     self.files = files_
        
        if ignore_files:
            self.files = list(set(self.files) - set(ignore_files))
        self.num_files = len(self.files)
        if not silent:
            print(f"loaded {len(self.files)} files, ignoring {len(ignore_files)} files")
        
    def __getitem__(self, idx):

        if self.metadata_file:
            file = self.files[idx]
            metadata = self.metadata_df.loc[self.metadata_df.filename == file][self.metadata_columns].to_dict('r')
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
        return idx, file, text, token_count, metadata

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
                 tokenizer: Optional[GPT2Tokenizer] = None,
                 vectorizer = None,
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
        Domain dataset with document vectorization built in.
        """
        super().__init__(domain_directory=domain_directory,
                 filenames=filenames,
                 add_bos_token=add_bos_token,
                 bos_token=bos_token,
                 sample=sample,
                 sample_from_head=sample_from_head,
                 track_token_count=track_token_count,
                 anonymize=anonymize,
                 metadata_columns=metadata_columns,
                 ignore_files=ignore_files,
                 sample_by_metadata=sample_by_metadata,
                 metadata_file=metadata_file,
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
        tokenized_text = self.tokenizer(text)
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




def get_dataloader(domains, files=None, sample=None, num_workers=16, batch_size=16,metadata_file=None, metadata_columns=["domain"], silent=False):
    datasets = []
    for domain in tqdm(domains):
        resolved_path = Path(DATA_DIR) / domain / domain
        if files:
            domain_files = files
        else:
            with open(Path(DATA_DIR) / domain  / "metadata" / "filenames.txt", 'r') as f:
                domain_files = []
                for x in f.readlines():
                    fp = x.strip()
                    domain_files.append(fp)
            if sample:
                domain_files = np.random.choice(domain_files, sample)
        dataset = Domain(resolved_path,
                         filenames=list(domain_files),
                         add_bos_token=True,
                         metadata_file=metadata_file,
                         metadata_columns=metadata_columns,
                         silent=silent)
        datasets.append(dataset)
    
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size)
    return  dataloader
    
def read_domains(domains, files=None, sample=None, num_tokens=None, dataloader_only=False, metadata_file=None, metadata_columns = ["domain"], num_workers=16, batch_size=16, silent=False):
    
    dataloader = get_dataloader(domains, files, sample, batch_size=batch_size, num_workers=num_workers, metadata_file=metadata_file, metadata_columns=metadata_columns, silent=silent)
    
    if dataloader_only:
        return dataloader
    curr_tokens = 0
    texts = []
    files = []
    metadatas = []
    pbar = tqdm(dataloader)
    for id, file, text, token_count, metadata in pbar:
        for  f, t, tc in zip(file, text, token_count):
            curr_tokens += tc
            pbar.set_description(f"number of tokens: {humanize.intword(curr_tokens)}")
            if num_tokens and curr_tokens > num_tokens:
                break
            texts.append(t)
            files.append(f)
            metadatas.append(metadata)
        if num_tokens and curr_tokens > num_tokens:
            break

    df = pd.DataFrame({"text": texts,
                       "file": files,
                       "metadata": metadatas})
    df['id'] = range(len(df))
    return df

