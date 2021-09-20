import logging
from time import time
from typing import Dict, Any, Optional
import json
from collections import defaultdict
import numpy as np
from sklearn.cluster import (MiniBatchKMeans, KMeans)
# from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained   
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
sns.set_style('white')
sns.set_palette('colorblind')
import scipy


CLUSTERING_MODELS = {
        # "Agglomerative": AgglomerativeClustering,
        "KMeans": KMeans,
        "KMeansConstrained": KMeansConstrained,
        # "DBSCAN": DBSCAN,
        # "HDBSCAN": HDBSCAN,
        "MiniBatchKMeans": MiniBatchKMeans    }
 

def load_clusters(path_to_serialized_clusters: Path) -> pd.DataFrame:
    with open(path_to_serialized_clusters, 'r') as f:
        clusters = json.load(f)             
    logging.info(f"Clusters found, loading cluster from {path_to_serialized_clusters}...")
    clustering_output = {int(x): y for x,y in clusters['clusters'].items()}
    clusters = {'clusters': clustering_output}
    idxs = {}
    for k, vals in tqdm(clusters['clusters'].items()):
        for val in vals:
            idxs[val] = k
    return clusters, idxs

class Timer(object):
    def __init__(self, description, silent=False):
        self.silent = silent
        self.description = description
    def __enter__(self):
        self.start = time()
    def __exit__(self, type, value, traceback):
        self.end = time()
        if not self.silent:
            logging.info(f"{self.description}: {self.end - self.start} seconds")

class Cluster:
    def __init__(self, model_type: str, domain_label_map: Dict=None) -> None:
        """
        main clustering module
        Params
        ------
        model_type: str => type of model (see CLUSTERING_MODELS for options)
        domain_label_map: Dict (default None) => map of ground truth domain labels
        """
        self.model_type = model_type
        self.model = CLUSTERING_MODELS[model_type]
        self.domain_label_map = domain_label_map

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
        return pca, vectors
    
    def create_linkage_matrix(self, **kwargs):
        """
        this is only relevant for agglomerative clustering, see sklearn docs for this
        """
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_,
                                          counts]).astype(float)
        return linkage_matrix

    def calculate_purity(self, corpus, output):
        corpus.corpus['clusters'] = output['clustering']
        return corpus.groupby(['domain', 'cluster']).count()

    def cluster_vectors(self,
                        vectors: np.ndarray,
                        indices: np.ndarray,
                        output_dir : Optional[str]=None,
                        reduce_dimensions_first: bool=True,
                        num_components: Optional[int]=None,
                        silent: bool=False,
                        **kwargs) -> Dict[str, Any]:
        """
        cluster vectors!

        Params
        ------
        vectors: np.ndarray => vectors to cluster (output of load_embeddings)
        indices: np.ndarray => associated indices (output of load_embeddings)
        output_dir: str => path to directory where we will serialize clustering output
        reduce_dimensions_first: bool => if set, we will perform dimensionality reduction (PCA) on the embeddings
        num_components: int => number of dimensions to reduce to
        silent: bool => if set, will disable tqdm
        **kwargs => any additional keyword arguments to provide to  clustering algorithm
        """
        output = {}
        output['model'] = self.model(**kwargs)
        output['model_type'] = self.model_type
        if reduce_dimensions_first:
            assert num_components is not None
            print("reducing dimensions...")
            with Timer('Elapsed Time', silent):
                pca, vectors = self.reduce_dimensions(vectors, num_components)
            output['pca'] = pca
        print("clustering...")
        with Timer('Elapsed Time', silent):
            output['clustering'] = output['model'].fit_predict(vectors)
        output['clusters'] = defaultdict(list)
        for i, c in enumerate(output['clustering']):
            output['clusters'][int(c)].append(indices[i])
            
        if not silent:
            logging.info(f"Detected {len(output['clusters'])} clusters")
        # if self.model_type == 'HDBSCAN':
            # output['soft_clusters'] = hdbscan.all_points_membership_vectors(output['model'])
        elif self.model_type == 'Agglomerative' and kwargs.get('distance_threshold'):
            output['linkage_matrix'] = self.create_linkage_matrix()
        if output_dir is not None:
            output_dir_path = Path(output_dir)
            if not output_dir_path.exists():
                output_dir_path.mkdir()
            with open(output_dir_path / 'cluster_output.json', 'w+') as f:
                json.dump({"clusters": {x: list(y) for x, y in output['clusters'].items()}}, f)
        return output