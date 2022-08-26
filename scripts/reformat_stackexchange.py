from domain_loader.domain_loader import Domain
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
import jsonlines

import gzip
from typing import List, Dict
import json


def writeall_jsonl_gz(filename, payload: List[Dict], dumps=None):
    with gzip.open(filename, 'wb') as fp:
        json_writer = jsonlines.Writer(fp, dumps=dumps)
        json_writer.write_all(payload)



data_dir = Path("/private/home/suching/stackexchange-dataset/out/")

directories = data_dir.glob("*/")

for directory in tqdm(directories):
    texts = []
    domain = Domain(data_dir / directory)
    dataloader = DataLoader(domain,
                                num_workers=16,
                                batch_size=16)
    pbar = tqdm(dataloader)
    for id, file, text, token_count, _ in pbar:
        for f, t in zip(file, text):
            texts.append({"id": f, "text": t})

    writeall_jsonl_gz(data_dir / f"{directory}.jsonl.gz", texts, dumps=json.dumps)