import logging
from itertools import groupby
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import torch

class EmbeddingsProcessor:
    def __init__(self, directory: Path):
        """
        Core embeddings processor module.
        
        Params
        ------
        directory : Path => path to directory containing embeddings. 

        Assumes embeddings are sharded and grouped into vectors / ids by similar file prefix.

        """
        self.directory = directory
        files = list(self.directory.iterdir())
        files.sort()
        files = [file for file in files if str(file).endswith('.pt')]
        self.prefixes = files
        self.num_shards = len(self.prefixes)

    def iterate_across_mmap_shards(self, batch_size: int=None, sample: int=None):
        """
        Create a generator across embedding files to load them into memory.
        
        Params
        ------
        batch_size: int => batch size to use when loading each shard. If not set, will load entire shard at once.
        sample: int => if set, will subsample embeddings across shards (random sample)
        """
        for submat in tqdm(self.prefixes, desc="loading embeddings"):
            if str(submat).endswith('.emb.npy'):
                submat = np.load(submat)
                subids = np.load(subids)
            else:
                mat_ = torch.load(submat)
                subids = mat_[0].cpu().numpy()
                submat = mat_[1].cpu().numpy()
            if sample:
                if sample // self.num_shards < submat.shape[0]:
                    idx = np.random.choice(submat.shape[0], sample // self.num_shards, replace=False)
                    submat = submat[idx, :]
                    subids = subids[idx, :]
            if batch_size:
                if batch_size < submat.shape[0]:
                    mat_batches = np.array_split(submat, int(submat.shape[0] // batch_size))
                    id_batches = np.array_split(subids, int(subids.shape[0] // batch_size))
                else:
                    mat_batches = [submat]
                    id_batches = [subids]
                for mat_batch, id_batch in zip(mat_batches, id_batches):
                    yield mat_batch, id_batch
            else:
                yield submat, subids
