from cluster.cluster import Cluster, MMDataset, load_model
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
from fairseq.data.encoders.gpt2_bpe import get_encoder
import numpy as np
import torch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-to-vecs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--original-text-file", type=str)
    args = parser.parse_args()

    kmeans = Cluster('KMeans')

    # kmeans.save("model.pkl")
    # kmeans = kmeans.load("model.pkl")

    # batched_y = batchify(y, batch_size=16)
    # cluster_ids_y_ = []
    # for batch in tqdm(batched_y):
    #     output = kmeans.predict(
    #         X=batch,  distance='euclidean', balanced=False
    #     )
    #     cluster_ids_y_.append(output)
        


    dataset = MMDataset(args.path_to_vecs)

    kmeans.train(dataset, n_clusters=11, output_dir=args.output_dir)
    kmeans = load_model(args.output_dir + "/model.pkl")

    dataset = MMDataset(args.path_to_vecs)
    clusters = []
    for dataset_idx, ix, batch in DataLoader(dataset, batch_size=128):
        clusters.extend(list(zip(dataset_idx, ix, kmeans['model'].predict(batch.numpy()))))

    
    tmp_dict = []
    for item in clusters:
        tmp_dict.append({'dataset': item[0].item(), 'index': int(item[1]), 'cluster': item[2]})
    clusters = pd.DataFrame(tmp_dict)

    # output['clusters_1'] = {}
    # for key, value in output['clusters'].items():
    #     output['clusters_1'][key.item()] = value
    # output['clusters'] = output['clusters_1']
    # del output['clusters_1']
    # {"clusters": {x: list(y) for x, y in output['clusters'].items()}} 
    
    with open(args.original_text_file, 'r') as f:
        z = f.read()

    texts = [int(x.strip()) for x in z.split()]
    texts = list(tqdm(chunks(texts, 2048)))
    bpe = get_encoder("/private/home/suching/raw_data/demix_scale/data-bin/encoder.json",
                      "/private/home/suching/raw_data/demix_scale/data-bin/vocab.bpe")
    texts = pd.DataFrame([{'index': i, 
                            'text': bpe.decode(x)} for i, x in tqdm(enumerate(texts))])
    res = clusters.merge(texts, on='index')
    import pdb; pdb.set_trace()        

    # loader = DataLoader(dataset, batch_size=args.batch_size)
    # batches = []
    # features = []
    # for ix, batch in tqdm(loader):
    #     kmeans = kmeans.partial_fit(batch)
    # kmeans.save_model(args.serialization_dir)