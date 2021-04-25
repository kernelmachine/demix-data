import logging
from pathlib import Path
from typing import List, Union

import numpy as np
from sklearn.decomposition import PCA

from cluster.embeddings_processor import EmbeddingsProcessor


class Corpus:

    def __init__(self,
                 path_to_embeddings: Union[Path, List[Path]]=None,
                 path_to_corpus: Union[Path, List[Path]]=None): 
        """
        Core corpus module.

        Params
        ------
        path_to_embeddings: Path => path to directory containing embeddings
        path_to_corpus: Path => path to directory containing corpus
        """
        self.path_to_corpus = [path_to_corpus] if not isinstance(path_to_corpus, List) else path_to_corpus
        self.path_to_embeddings = [path_to_embeddings] if not isinstance(path_to_embeddings, List) else path_to_embeddings
        if self.path_to_embeddings:
            self.file_iterators = [EmbeddingsProcessor(path) for path in self.path_to_embeddings]
        self.corpus = None
        

    def reduce_dimensions(self, vectors: np.ndarray, num_components: int) -> np.ndarray:
        """
        reduce dimensions. currently PCA is default, but may change?
        Params
        ------
        vectors: np.ndarray => vectors to reduce dimensions of
        num_components: int => number of dimensions to reduce to
        """
        pca = PCA(n_components=num_components)
        vectors = pca.fit_transform(vectors)
        return vectors

    def load_embeddings(self, sample: int=None, reduce_dimensions: bool=True, dim: int=10) -> None:
        """
        load embeddings from corpus
        
        Params
        ------
        sample: int => if set, we will only load the provided number of embeddings (random sample)
        reduce_dimensions: bool => if set, reduce dimensions after loading (using PCA)
        dim: bool => target dimensions to reduce to
        """
        vectors = []
        indices = []
        for file_iterator in self.file_iterators:
            for vecs, ids in file_iterator.iterate_across_mmap_shards(64, sample=sample):
                vectors.append(vecs)
                indices.append(ids)
    
        self.vectors = np.concatenate(vectors, 0)
        self.indices = np.concatenate(indices, 0)
        if reduce_dimensions and (self.vectors.shape[1] != dim):
            logging.info(f"reducing dimensions of embeddings to {dim}...")
            self.vectors = self.reduce_dimensions(self.vectors, dim)
        self.dim = self.vectors.shape[1]

    def take_subset_of_embeddings(self, idxs: List[int]) -> None:
        """
        Take a subset of loaded embeddings with provided index. You might do this if you 
        load only a sample of a corpus, and want embeddings only for that sample.

        Params
        ------
        idxs: List[int] => list of indices corresponding to embeddings you would like
        """
        self.indices = self.indices[idxs]
        self.vectors = self.vectors[idxs, :]
    
    def load_corpus(self, batch_size: int=None, sample: int=None, ignore_idx: List[int]=None) -> None:
        raise NotImplementedError
