import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from mode.cluster.corpus import Corpus
from mode.cluster.cluster import Cluster, load_clusters
# import networkx
# from networkx.convert_matrix import from_numpy_matrix
import torch
from pathlib import Path

sns.set_style('white')

def get_mode(list_):
    return (pd.Series(list_).value_counts(normalize=True) * 100) .to_dict()

domain_label_map = {
        0: "1b",
        1: "realnews",
        2: "legal",
        3: "tweets",
        4: "biomed",
        5: "cs",
        6: "reviews"
    }

class Graph:
    def __init__(self, corpus: Corpus):
        """
        Module for analyses + plots
        Params
        ------
        corpus: pd.DataFrame => loaded corpus with text
        clustering_output: Dict => output of clustering algorithm (return value of cluster.Cluster)
        """
        self.corpus = corpus
        self.domain_embeddings = None
        self.adjacency_matrix = None
        self.graph = None
        self.domains = None
        self.domain_probs = None
    
    def get_domain_embeddings(self, path_to_corpus=None, domains=None, cluster=False, path_to_clusters=None, n_clusters=8):
        def get_domain_embedding(group):
            docs = group.sample(n=min(group.shape[0], num))
            return np.expand_dims(np.mean(self.corpus.vectors[docs.id], axis=0), axis=0)
        tqdm.pandas()
        
        if cluster:
            domain_embeddings = []
            domains = []

            if path_to_clusters is None:
                clustering = Cluster("MiniBatchKMeans")
                clusters = clustering.cluster_vectors(self.corpus.vectors,
                                self.corpus.indices,
                                n_clusters=n_clusters,
                                reduce_dimensions_first=False,
                                random_state=0)
            # clusters_path = Path(path_to_corpus) / 'cluster_output.json'
            # clusters_path = Path("/gscratch/zlab/sg01/1b_tweets_legal_realnews_cs_reviews_biomed_webtext/1b_tweets_legal_realnews_cs_reviews_biomed_webtext_0.125_0.125_0.125_0.125_0.125_0.125_0.125_0.125/") / 'cluster_output.json'
            else:
                clusters, _ = load_clusters(path_to_clusters)
            self.domain_token_counts = {}
            for cluster_id, cluster in clusters['clusters'].items():
                domains.append(cluster_id)
                domain_embeddings.append(np.expand_dims(np.mean(self.corpus.vectors[list(cluster), :], axis=0), axis=0))
                self.domain_token_counts[cluster_id] = self.corpus.corpus.loc[self.corpus.corpus.id.isin(cluster)].text.apply(lambda x: len(x.split())).sum()
            self.domains = domains
            self.domain_embeddings = torch.tensor(np.concatenate(domain_embeddings, 0)).float().cuda()
            
        else:
            if domains:
                corpus = self.corpus.corpus.loc[self.corpus.corpus.domain.isin(domains)]
            else:
                corpus = self.corpus.corpus

            mask = torch.triu(torch.ones(self.corpus.vectors.shape[1], self.corpus.vectors.shape[1])).transpose(0, 1).bool()

            domain_embeddings = []
            ds = list(corpus.groupby('domain'))
            ds = sorted(ds, key=lambda x: x[0])

            for ix, (name, group) in tqdm(enumerate(ds), total=len(ds)):
                means = np.zeros((group.shape[0], mask.shape[0], self.corpus.vectors.shape[2]))
                subvecs = self.corpus.vectors[group.id, : , :]
                for i in trange(mask.shape[0], leave=False):
                    means[:,i,:] = (np.mean(subvecs[:, mask[i,:] , :], 1))
                domain_embeddings.append(np.expand_dims(means, 0))
            domain_embeddings = np.concatenate(domain_embeddings, 0)
            self.domain_embeddings = domain_embeddings.mean(1)
            self.domains = [x for x,y in ds]
            # out = corpus.groupby('domain').progress_apply(get_domain_embedding).sort_index()
            # self.domains = out.index.values
            # if torch.cuda.is_available():
            #     self.domain_embeddings = torch.tensor(np.concatenate(out.values, 0)).float().cuda()
            # else:
            #     self.domain_embeddings = torch.tensor(np.concatenate(out.values, 0)).float()
    
    def compute_cosine_similarities(self):
        # base similarity matrix (all dot products)
        # replace this with A.dot(A.T).toarray() for sparse representation
        similarity = np.dot(self.domain_embeddings, self.domain_embeddings.T)

        # squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)

        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        self.adjacency_matrix = cosine.T * inv_mag
    
    def build_graph(self, domains=None):
        self.get_domain_embeddings(domains)
        self.compute_cosine_similarities()
        self.graph = from_numpy_matrix(self.adjacency_matrix, create_using=networkx.Graph)
        self.graph = networkx.relabel_nodes(self.graph, {x: y for x,y in enumerate(self.domains)})

    def get_neighbors(self, domain: str, min_threshold=0.0):
        """
        Get neighbors of a subreddit with similarity above min_threshold.
        """
        list_neighbors = networkx.neighbors(self.graph, domain)
        res = []
        for i in list_neighbors:
                if self.graph.edges[(domain,i)]['weight'] > min_threshold:
                    res.append({"domain": i, **self.graph.edges[(domain,i)]})
        return pd.DataFrame(res)
            
    def prune_edges(self, min_threshold=0.0):
        """
        Prune the edges of a graph to edges with weight above min_threshold
        """
        edge_weights = networkx.get_edge_attributes(self.graph,'weight')
        self.graph.remove_edges_from((e for e, w in edge_weights.items() if w < min_threshold))

    def get_domain_prob_vec(self, vectors):
        distances = torch.cdist(domain_embeddings, vectors, p=2)
        distances =  torch.nn.functional.softmax(-distances, dim=-1)
        return distances 


    def get_domain_probs(self):
        batches = np.array_split(self.corpus.vectors, self.corpus.vectors.shape[0] // 128, axis=0)
        distances = []
        for batch in batches:
            distances.append(torch.norm(self.domain_embeddings - torch.tensor(batches[0][:, None, :]).cuda(), dim=-1))
        distances = torch.cat(distances, 0)
        self.domain_probs = torch.nn.functional.softmax(-distances, dim=-1)

    def generate_soft_domains(self):
        domain_map = {x: y for x,y in enumerate(self.domains)}
        batches = np.array_split(self.corpus.vectors, self.corpus.vectors.shape[0] // 128, axis=0)
        distances = []
        for batch in tqdm(batches):
            distances.append(torch.norm(self.domain_embeddings - torch.tensor(batch[:, None, :]).cuda(), dim=-1).argmax(1))
        distances = torch.cat(distances, 0)
        domains = pd.Series(distances.squeeze(0).tolist())
        domains = domains.apply(lambda x : domain_map[x])
        self.corpus.corpus['domain'] = domains