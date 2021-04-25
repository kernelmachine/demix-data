from cluster.mixed_corpus import Mixed
from pathlib import Path
import glob
import numpy as np
from copy import copy
import random
import pandas as pd
import uuid
from cluster.cluster import Cluster


def concatenate_corpora(corpora):
    master = copy(corpora[0])
    dfs = [corpus.corpus for corpus in corpora]
    master.corpus = pd.concat(dfs, 0)
    master.vectors = np.concatenate([corpus.vectors for corpus in corpora], 0)
    indexes = []
    for corpus in corpora:
        indexes.append(corpus.indices)
    master.indices = np.array([x for y in indexes for x in y])
    return master 
    
data_path = Path('../data_proc/data/')
domains = ['legal', 'cs', 'reviews', 'realnews']
mixed_corpora = []
for ix, domain  in enumerate(domains):
    
    if  (data_path / domain / f'{domain}.train.jsonl').exists(): 
        print(data_path / domain / f'{domain}.train.jsonl')
        corpus = Mixed(path_to_corpus=[data_path / domain / f'{domain}.train.jsonl'],
            path_to_embeddings=data_path / domain / f'{domain}_embeddings')
        corpus.load_embeddings(reduce_dimensions=False)
        corpus.load_corpus()
        corpus.corpus['domain'] = domain
        # map_ = {x.item(): uuid.uuid4().hex for x in corpus.indices}
        # corpus.corpus.id = corpus.corpus.id.apply(lambda x: map_[x])
        # corpus.indices = [map_[x.item()] for x in corpus.indices]
        mixed_corpora.append(corpus)

master = concatenate_corpora(mixed_corpora)
cluster = Cluster('MiniBatchKMeans')
output = cluster.cluster_vectors(master.vectors, master.indices, reduce_dimensions_first=False, n_clusters=8)
cluster.calculate_purity(master, output)
from fairseq import pdb; pdb.set_trace()
