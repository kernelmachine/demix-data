from cluster.analysis import ClusterAnalysis
from cluster.cluster import load_clusters, Cluster
import pandas as pd
from domain_loader.split_subdomains import compute_tfidf
from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS



if __name__ == "__main__":
    
    print('reading data...')
    meta = pd.read_json(PROJECT_DIR / "openwebtext" / "metadata" / "metadata.1.jsonl", lines=True)
    meta['id'] = range(len(meta ))
    print('done')
    
    tfidf_vecs, metadata = compute_tfidf(meta, "openwebtext")

    clustering = Cluster('KMeansConstrained')
    indices = range(tfidf_vecs.shape[0])
    subdomains = clustering.cluster_vectors(vectors=tfidf_vecs, 
                                            indices=indices,
                                            n_clusters=8,
                                            size_min=50,
                                            reduce_dimensions_first=True,
                                            num_components=50,
                                            output_dir=None)
    analysis = ClusterAnalysis(meta,subdomains)
    analysis.plot_tsne(tfidf_vecs, save_path="tsne.pdf")