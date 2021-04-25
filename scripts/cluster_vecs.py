from cluster.cluster import Cluster
import numpy as np


if __name__ == '__main__':
    kmeans = Cluster("KMeansConstrained")
    data = np.load('test.npy')
    data[np.isnan(data)] = 0
    data /= 100
    data = 1 - data
    labels = np.load('test.npy.labels.npy')
    indices = list(range(len(data)))
    clustering = kmeans.cluster_vectors(data, indices, n_clusters=8, size_min=50, reduce_dimensions_first=False, output_dir="./")
    import pdb; pdb.set_trace()