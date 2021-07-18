import os
from typing import Optional, List, Tuple
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
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from scipy import sparse
import re

def vec_collate_fn(batch):
        return [x[0] for x in batch],  sparse.vstack([x[1] for x in batch]),  [x[2] for x in batch]


def reservoir_sampling( iterator, K ):
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


def collate_fn(batch):
    return [x[0] for x in batch],  [x[1] for x in batch],  [x[2] for x in batch]








class Domain(Dataset):
    def __init__(self,
                 domain_directory: Path,
                 filenames: Optional[List[str]] = None,
                 add_bos_token: bool = False,
                 metadata_columns: List[str] = None,
                 ignore_files: Optional[List[str]] = [],
                 sample_by_metadata: Optional[Tuple[str, int]] = None,
                 metadata_file: Optional[Path] = None,
                 sample: int = None,
                 sample_from_head: bool = False,
                 track_token_count: bool = False,
                 anonymize: bool = False,
                 **metadata_filters):
        super().__init__()      
        self.add_bos_token = add_bos_token
        self.anonymize = anonymize 

        re1 = {
            "regex": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
            "repl": "<|EMAIL|>"
        }

        re2 = {"regex": "[0-9]{10}-[0-9A-Fa-f]{53}", "repl":"<|DART|>"}

        re3 = {"regex": "@\[[0-9]+:[0-9]+:(?=[^\]])(([^\\:\]]*(?:\\.)*)*)\]", "repl": "<|FBUSERID|>"}

        re4 ={"regex": "(?:(?<![\d-])(?:\+\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-\/.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))", "repl": "<|PHONE_NUMBER|>"}

        re5 ={"regex": "(?:4\d{12}(?:\d{3})?|(?:5[1-5]\d{2}|222[1-9]|22[3-9]\d|2[3-6]\d{2}|27[01]\d|2720)\d{12}|3[47]\d{13}|5019\d{12}|3(?:0[0-5]|[68]\d)\d{11}|6(?:011|5\d{2})\d{12}|(?:2131|1800|35\d{3})\d{11})", "repl": "<CREDIT_CARD_NUMBER>"}

        re6 ={"regex": "(?!(?:000|666|9))\d{3}-(?!00)\d{2}-(?!0000)\d{4}", "repl": "<|SSN|>"}

        re7 = {"regex": "\d+\s(?:(?:[a-z0-9.-]+[ ]?)+\s(?:Alley|Aly|Ave(?:nue)?|Boulevard|Blvd|Br(?:anch)?|Center|Ctr|Cir(?:cle)?|Court|Ct|Crossing|Xing|Dr(?:ive)?|Est(?:ate)?|Expressway|Expy|Freeway|Fwy|Highway|Hwy|Hills|Hls|Knoll|Knl|Landing|Lndg|Lane|Ln|Manor|Mnr|Meadow|Mdw|Parkway|Pkwy|Pass|Path|Plaza|Plz|Road|Rd|Run|Sq(?:uare)?|St(?:ation|reet|a)?|Ter(?:ace)?|Trail|Trl|Turnpike|Tpke|Valley|Vly|View|Vw|Village|Vlg|Vis(?:ta)?|Walk|Way)|(?:Route|Rte|Interstate|I)[- ]?\d{1,3})(?:\s(?:Apt[\.]?|Apartment|#)[ ]?\d+[a-z]?)?(?:\s(?:[a-z-]+[ ]?)+,?(?:\s(?:AK|AL(?:aska|abama)?|AR(?:kansas|izona)?|AZ|CA(?:lifornia)?|CO(?:lorado|nnecticut)?|CT|DC|DE(?:laware)?|FL(?:orida)?|GA|Georgia|GU(?:am)?|HI|Hawaii|IA|Iowa|ID(?:aho)?|IL(?:linois)?|IN(?:diana)?|KS|Kansas|KY|Kentucky|LA|Louisiana|MA(?:ssachusetts|ryland|ine)?|MD|ME|MI(?:chigan|nnesota|ssissippi|ssouri)|MN|MO(?:ntana)?|MS|MT|NC|North[ ]Carolina|ND|North[ ]Dakota|NH|New[ ]Hampshire|NJ|New[ ]Jersey|NM|New[ ]Mexico|NV|Nevada|NY|New[ ]York|OH(?:io)?|OK(?:lahoma)?|OR(?:egon)?|PA|Pennsylvania|PR|Puerto[ ]Rico|RI|Rhode[ ]Island|SC|South[ ]Carolina|SD|South[ ]Dakota|TN|Tennessee|TX|Texas|UT(?:ah)?|VA|Virginia|VI(?:rgin[ ]Islands)?|VT|Vermont|WA(?:shington(?:[ ]D[. ]?C[.]?)?)?|WI(?:sconsin)?|WV|West[ ]Virginia|WY(?:oming)?)(?:\s\b\d{5}(?:-\d{4})?\b)?)?)?", 
            "repl": "<|ADDRESS|>"}

        re8 = {"regex": "@[a-zA-Z0-9_\.\-]{1,30}", "repl": "@USER"}


        re_list = [re1,re2,re3,re4,re5,re6,re8]


        self.anonymizer = {re.compile(x['regex']): x['repl'] for x in re_list}



        self.bos_token = "<|endoftext|> " 
        self.domain_directory = domain_directory
        self.files = {}
        if sample_by_metadata:
            self.metadata_counts = defaultdict(int)

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
                for file in self.files:
                    if self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] < sample_by_metadata['sample_size']:
                        self.metadata_counts[self.files[file][sample_by_metadata['metadata_column']]] += 1
                        files_[file] = self.files[file]
                self.files = files_
        else:
            if filenames:
                if isinstance(filenames[0], Tuple):
                    self.files = dict(filenames)
                else:
                    self.files = filenames
                # assert all(file.exists for file in self.files)
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
                # if sample is not None:
                #     print(f"Loading {sample} of files from {domain_directory}...")
                #     for ix, x in enumerate(fs):
                #         if ix < sample:
                #             self.files[Path(x)] = []
                #         else:
                #             break
                # else:
        
        # self.files_  = {}
        # if sample is not None:
        #     print(f"{sample} files from {domain_directory}...")
        #     for ix, x in enumerate(self.files):
        #         if ix < sample:
        #             self.files.append(Path(fname))
        #         else:
        #             break
        #     else:

        # if metadata_filters:
        #     if metadata_file is None:
        #         raise Exception("metadata_file cannot be none if metadata_filters applied.")
        #     print("metadata filter detected, loading metadata...")
        #     metadata = pd.read_json(metadata_file, lines=True)
            
        #     for filename in self.filenames:
        #         if 
        #     fnames = []
        #     for key, items in metadata_filters.items():
        #         m = metadata.loc[metadata[key].isin(items)]
        #         if 'filename' in m.columns:
        #             m.filename.apply(lambda x: fnames.extend(x))
        #         else:
        #             fnames.extend(m.filename.values)
        #     filenames = list(set(fnames))
        #     filenames = [domain_directory / Path(filename) for filename in filenames]

        # if sample_by_metadata:
        #     self.sample_by_metadata = True
        #     if metadata_file is None:
        #         raise Exception("metadata_file cannot be none if sample_by_metadata applied.")
        #     print("metadata sampling detected, loading metadata...")
            
        #     filenames = defaultdict(list)
        #     # metadata = pd.read_json(metadata_file, lines=True)
            
        #     # [sample_by_metadata['metadata_column']].value_counts()
        #     # most_prevalent = most_prevalent.loc[most_prevalent > 1000]
        #     # most_prevalent = most_prevalent.index.values
    
        #     with open(metadata_file, 'r') as f:
        #         for line in tqdm(f):
        #             if len(filenames[z[sample_by_metadata['metadata_column']]]) < sample_by_metadata['sample_size']:
        #                 fname = z['filename']
        #                 filenames[z[sample_by_metadata['metadata_column']]].append(fname)
            
        #     filenames = list(filenames.values())
        #     print(f"loaded {len(filenames)} files.")
        #     from fairseq import pdb; pdb.set._trace()
        # else:
        #     self.sample_by_metadata = False
        
        # if self.filenames:
        #     if sample is not None:
        #         self.files = []
        #         print(f"Loading {sample} of files from {domain_directory}...")
        #         for ix, (fname, x) in enumerate(self.filenames.items()):
        #             if ix < sample:
        #                 self.files.append(Path(fname))
        #             else:
        #                 break
        #     else:
        #         print(f"Loading list of files from {domain_directory}...")
        #         self.files = [Path(x) for x in self.filenames]
        #     # if self.sample_by_metadata:
        #     #     assert all(all(file.exists for file in y) for _, y in self.files)
        #     # else:
        #     assert all(file.exists for file in self.files)
        # else:
        #     fs = tqdm(domain_directory.glob("*/*"))
        #     if sample is not None:
        #         self.files = []
        #         print(f"Loading {sample} of files from {domain_directory}...")
        #         for ix, x in enumerate(fs):
        #             if ix < sample:
        #                 self.files.append(Path(x))
        #             else:
        #                 break
        #     else:
        #         print(f"Loading all files from {domain_directory}...")
        #         self.files = list(fs)
        
        if ignore_files:
            self.files = list(set(self.files) - set(ignore_files))
            # for file in ignore_files:
            #     if self.files.get(file):
            #         del self.files[file]
            # self.files = set([str(x) for x in self.files]) - set(ignore_files)
            # self.files = [Path(x) for x in self.files]
        print(f"loaded {len(self.files)} files, ignoring {len(ignore_files)} files") 
        # self.files = list(self.files.items())
    def __getitem__(self, idx):
        # if self.sample_by_metadata:
        #     metadata, files = self.files[idx]
            
        #     files = [file for file in files]
        #     texts = []
        #     for file in files:
        #         if file.name.endswith('.gz'):
        #             with gzip.open(file, 'rb') as f:
        #                 text = f.read().decode('utf-8')
        #         else:
        #             text = file.read_text(errors='ignore')
        #         if self.add_bos_token:
        #             text = self.bos_token + text
        #         texts.append(text)
        #     return [str(x) for x in files], texts, [metadata] * len(files)
        # else:
        
        file = str(self.files[idx])
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
            text = self.bos_token + text
        if self.anonymize:
            for x,y in self.anonymizer.items():
                text = x.sub(y, text)
        token_count = len(text.split())
        return file, text, token_count, []

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
        super().__init__(domain_directory=domain_directory,
                        metadata_file=metadata_file,
                        filenames=filenames,
                        ignore_files=ignore_files,
                        **metadata_filters)
        if not tokenizer:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer = tokenizer

    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text = super().__getitem__(idx)
        return filename, np.array(self.tokenizer.encode(text, truncation=True))


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

        # def partial_fit(file, vectorizer):
        #     file = self.domain_directory / file
        #     if file.name.endswith('.gz'):
        #         with gzip.open(file, 'rb') as f:
        #             text = f.read().decode('utf-8')
        #     else:
        #         text = file.read_text(errors='ignore')
        #     vectorizer.fit([text])
        # for file, _ in tqdm(self.files):
        #     partial_fit(file, self.vectorizer)
            
    def __getitem__(self, idx) -> Tuple[str, np.array]:
        filename, text, metadata = super().__getitem__(idx)
        tokenized_text = " ".join(map(str, self.tokenizer.encode(text, truncation=True)))
        return filename, tokenized_text, metadata


def domain_dataloader(domain_directory: Path,
                      metadata_file: Optional[Path] = None,
                      filenames: Optional[List[str]] = None,
                      tokenized=False,
                      batch_size=64,
                      **metadata_filters):

    if tokenized:
        dataset = DomainTokenized(domain_directory, metadata_file, filenames, **metadata_filters)
    else:
        dataset = Domain(domain_directory, metadata_file, filenames, **metadata_filters)
    dataloader = DataLoader(dataset, num_workers=os.cpu_count(), batch_size=batch_size)
    return dataloader


