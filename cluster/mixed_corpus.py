import pandas as pd
from tqdm.auto import tqdm

from cluster.corpus import Corpus
from typing import List

class Mixed(Corpus):
    def __init__(self, **kwargs):
        """
        Mixed Corpus Module

        Params
        ------
        db_username: str => postgres database username
        db_name: str => postgres database name
        """
        super().__init__(**kwargs)
    
    def load_corpus(self, batch_size: int=None, sample: int=None, ignore_idx: List[int]=None) -> None:
        """
        Load mixed corpus from database into memory
        
        Params
        ------
        batch_size: int => batch size to use when loading the corpus. If not set, will load entire corpus at once.
        sample: int => if set, will subsample corpus to provided size (random sample)
        """
        dfs = []
        for path in self.path_to_corpus:
            dfs.append(pd.read_json(path, lines=True))
        self.corpus = pd.concat(dfs, 0)
        if sample is not None:
            self.corpus = self.corpus.sample(n=sample)
        if ignore_idx is not None:
            self.corpus = self.corpus.loc[~self.corpus.id.isin(ignore_idx)]
        return
