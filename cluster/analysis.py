import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

sns.set_style('white')
sns.set_palette('colorblind')
def get_mode(list_):
    return (pd.Series(list_).value_counts(normalize=True) * 100) .to_dict()

domain_label_map = {
        0: "1b",
        1: "realnews",
        2: "legal",
        3: "reddit",
        4: "med",
        5: "cs",
        6: "reviews",
        7: 'openwebtext'
}

label_domain_map = {y: x for x,y in domain_label_map.items()}
class ClusterAnalysis:
    def __init__(self, corpus: pd.DataFrame, clustering_output: Dict):
        """
        Module for analyses + plots
        Params
        ------
        corpus: pd.DataFrame => loaded corpus with text
        clustering_output: Dict => output of clustering algorithm (return value of cluster.Cluster)
        """
        self.corpus = corpus.sort_values(by='id')
        self.clustering_output = clustering_output
    
    def get_cluster(self, cluster_idx: int) -> pd.DataFrame:
        return self.corpus.loc[self.corpus.id.isin(self.clustering_output['clusters'][cluster_idx])]

    def cluster_label_distributions(self, label: str='subreddit', cluster_idxs: List[int]=None) -> pd.DataFrame:
        """
        Get the label distributions per cluster
        Params
        ------
        label : str => label to analyze, e.g. "subreddit"/"url"/"domain"
        """
        clusters = self.clustering_output['clusters']
        if cluster_idxs:
            clusters_as_tuples = sorted([ (cluster_idx, example_idxs) for cluster_idx, example_idxs in clusters.items() if cluster_idx in cluster_idxs])
        else:
            clusters_as_tuples = sorted([ (cluster_idx, example_idxs) for cluster_idx, example_idxs in clusters.items()])
        distributions = []
        for c_idx, e_idxs in tqdm(clusters_as_tuples):
            labels = self.corpus.loc[self.corpus.id.isin(e_idxs)][label].values
            res = get_mode(labels)
            res = {domain_label_map[x]: y for x,y in res.items()}
            res['num_examples'] = len(e_idxs)
            res['cluster_idx'] = c_idx
            distributions.append(res)
        distributions_df = pd.DataFrame(distributions)
        entropies = []
        for ix, row in distributions_df.iterrows():
            entropies.append(stats.entropy(row.dropna() / 100))
        distributions_df['entropy'] = entropies
        return distributions_df
    
    def get_cross_documents(self, label: str, ground_truth_label: str, cluster_idx: int):
        return self.corpus.loc[(self.corpus.id.isin(self.clustering_output['clusters'][cluster_idx])) & (self.corpus.domain == label_domain_map[ground_truth_label])]

    def get_cluster_labels(self, cluster_idx: int, label: str='subreddit', num_labels: int=None) -> pd.DataFrame:
        """
        Get the labels of a particular cluster
        Params
        ------
        cluster_idx : int => index of cluster
        label: str => label to retrieve (e.g. "subreddit", "url", "domain")
        num_groups: int (default None) => number of labels to print 
        """
        return self.corpus.loc[self.corpus.id.isin(self.clustering_output['clusters'][cluster_idx])][label].value_counts().head(num_labels)
      
    def get_entropy_cluster_size(self, label: str='subreddit', cluster_idxs: List[int]=None, logx: bool=False, plot: bool=False):
        """
        Scatterplot of entropy vs number of labels of each cluster in corpus.
        Entropy approximates the variance of labels in a cluster.
        Params
        ------
        label: str => label to retrieve (e.g. "subreddit", "url", "domain")
        logx: bool => set x-axis as logscale?
        plot: bool => plot distribution?
        """
        distributions = self.cluster_label_distributions(label, cluster_idxs)
        if plot:
            ax = sns.scatterplot(data=distributions, x='num_examples', y='entropy')
            if logx:
                ax.set_xscale('log')
            ax.set_xlabel('number of examples in cluster')
            ax.set_ylabel('entropy of cluster')
        return distributions            
    
    def correlate_perplexity_cluster_features(self, perplexities_file):
        df = pd.read_json(perplexities_file, lines=True)
        entropy_cluster_size = self.get_entropy_cluster_size('subreddit')
        entropy_cluster_size['cluster'] = entropy_cluster_size.cluster.astype(int)
        df = df.merge(entropy_cluster_size)
        df['normalized'] = df.num_examples / df.entropy
        # df = df.loc[df.perplexity <50] remove outliers??
        sns.regplot(data=df, x='normalized', y='perplexity')
        return df[['normalized', 'perplexity']].corr()
    
    def plot_tsne(self,
                  data: np.ndarray,
                  disable_tqdm: bool=True,
                  sample: int=None,
                  legend: bool=True,
                  ground_truth: List[int]=None,
                  save_path: str = None):
        """
        Plot TSNE of embeddings

        Params
        ------
        data: np.ndarray => embeddings to plot TSNE of (e.g. `mixed_corpus.vectors`)
        disable_tqdm: bool => disable TQDM when plotting
        sample: int (default None) => size of subset of embeddings to plot
        legend: bool => display legend
        ground_truth: List[int] => ground truth cluster labels
        """
        if ground_truth is not None:
            ground_truth = list(self.corpus[ground_truth].values)
            ground_truth = np.array([x if x is not None else "None" for x in ground_truth])
        
        if sample is not None:
            idx = np.random.choice(data.shape[0], sample, replace=False)
            data = data[idx, :]
            if ground_truth is not None:
                ground_truth = ground_truth[idx]
            model_labels_ = self.clustering_output['model'].labels_[idx]
        else:
            model_labels_ = self.clustering_output['model'].labels_
        
        projection = TSNE(random_state=42, verbose=1).fit_transform(data)
        color_palette = sns.color_palette('Paired', max(np.unique(model_labels_)+1))
        if ground_truth is not None:
            ground_truth_color_palette = sns.color_palette('Paired', max(np.unique(model_labels_)+1))
        print(len(ground_truth_color_palette))
        if ground_truth is not None:
            fig, ax = plt.subplots(1,2, figsize=(20,8))
        else:
            fig, ax = plt.subplots(1,1, figsize=(8,8))
        
        cluster_colors = [color_palette[x] for x in model_labels_]
        
        if ground_truth is not None:
            cluster_colors_ground_truth = [color_palette[x] 
                                          for x in ground_truth]

        if self.clustering_output['model'] and 'probabilities_' in self.clustering_output['model'].__dict__:       
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, self.clustering_output['model'].probabilities_)]
        else:
            cluster_member_colors = cluster_colors
        
        unique_labels = np.unique(model_labels_)
        
        cluster_member_colors = np.array(cluster_member_colors)
        
        for label in tqdm(unique_labels, disable=disable_tqdm):
            idx = np.where(model_labels_ == label)[0]
            if ground_truth is not None:
                axes = ax[0]
            else:
                axes = ax
            axes.scatter(*projection.T[:, idx],
                         s=50,
                         linewidth=0,
                         c=cluster_member_colors[idx],
                         label=label)
            if legend:
                axes.legend(title='Cluster Label')
            axes.set_title(self.clustering_output['model_type'])
        
        if ground_truth is not None:
            unique_labels = np.unique(ground_truth)
            cluster_colors_ground_truth = np.array(cluster_colors_ground_truth)
            for label in tqdm(unique_labels, disable=disable_tqdm):
                idx = np.where(ground_truth == label)[0]
                ax[1].scatter(*projection.T[:, idx],
                              s=50,
                              linewidth=0,
                              c=cluster_colors_ground_truth[idx],
                              label=domain_label_map[label])
            ax[1].legend(title='Domain Label')
            ax[1].set_title('Domain Labels')
        if save_path is not None:
            plt.savefig(save_path, dpi=300)