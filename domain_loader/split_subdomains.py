import os
from typing import Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import GPT2Tokenizer
import pandas as pd
from pathlib import Path
import gzip
import random
from collections import defaultdict
import itertools
from domain_loader.constants import PROJECT_DIR, TOKEN_COUNTS
from domain_loader.utils import take_n_tokens
from tqdm.auto import tqdm
import numpy as np
from domain_loader.domain_loader import Domain, collate_fn
import argparse
import humanize
from cluster.cluster import load_clusters
from cluster.cluster import Cluster
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy import sparse


def compute_tfidf(meta ,  domain):
    print('reading file...')
    
    populated_domains = meta.domain.value_counts()
    populated_domains  = populated_domains.loc[populated_domains > 1000]
    populated_domains = populated_domains.index
    print('done')

    dataset = Domain(PROJECT_DIR / domain / domain,
                    filenames=None,
                    add_bos_token=False,
                    ignore_files=[],
                    metadata_columns=["domain"],
                    metadata_file=PROJECT_DIR / domain / "metadata" / "metadata.1.jsonl",
                    sample_by_metadata={"metadata_column": "domain", "sample_size": 10},
                    domain=populated_domains)
    

    loader = DataLoader(dataset,
                        num_workers=16,
                        batch_size=16,
                        collate_fn=collate_fn)

    pbar = tqdm(loader)
    # vectorizer = CountVectorizer(min_df=3, max_features=10000, stop_words="english", ngram_range=(2,2))
    ms = []
    vs = []
    
    docs = []
    for fname, text, metadata in pbar:
        for t,m in zip(text, metadata):
            docs.append({'domain': m['domain'], 'text': t})

    df = pd.DataFrame(docs)
    df['id'] = range(len(df))
    
    # dataset = Domain(PROJECT_DIR / domain / domain,
    #                 add_bos_token=False,
    #                 filenames=dataset.files)
    

    # loader = DataLoader(dataset,
    #                     num_workers=16,
    #                     batch_size=16,
    #                     collate_fn=collate_fn)
    count_vectorizer = CountVectorizer(stop_words="english")

    embeddings = count_vectorizer.fit_transform(tqdm(df['text']))

    def get_domain_embeddings(group):
        return np.mean(embeddings[group.id], axis=0)

    
    tqdm.pandas()
    out = df.groupby('domain').progress_apply(get_domain_embeddings).sort_index()
    domain_embeddings = np.concatenate(out.values, 0)
    domains = out.index.values
    # res = []
    # mat = []
    # for metadata, vecs in domain_embeddings.items():
    #     res.append(metadata)
    #     mat.append(vecs)
    # mat = np.concatenate(mat, 0)
    # file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    
    # overlaps = {}
    # for x, y in tqdm(file_pairs):
    #     intersection = vocabs[x] & vocabs[y]
    #     union = (vocabs[x] | vocabs[y])
    #     overlaps[x + "_" + y] = len(intersection) / len(union)
    
    # data = []

    # z = {}
    # for key in tqdm(overlaps.keys()):
    #     file_1, file_2 = key.split('_')
    #     if not z.get(file_1):
    #         z[file_1] = {}
    #     z[file_1][file_2] = overlaps[key]
    #     if not z.get(file_2):
    #         z[file_2] = {}
    #     z[file_2][file_1] = overlaps[key]

    # labels = list(vocabs.keys())

    # for ix, key in tqdm(enumerate(z)):
    #     items = []
    #     for subkey in labels:
    #         if not z[key].get(subkey):
    #             items.append(1.0)
    #         else:
    #             items.append(z[key][subkey])
    #     data.append(items)
    
    # data = np.array(data)
    # data[np.isnan(data)] = 0
    # data = 1 - data
    # import pdb; pdb.set_trace()
    # # pbar = tqdm(loader)
    # # for fname, text, metadata in pbar:
    # #     vs.append(vectorizer.transform(text))
    # #     ms.extend([m['domain'] for m in metadata])
    
    # vs_dict = defaultdict(list)
    # for m, v in zip(ms, vs):
    #     vs_dict[m].append(v)

    # res = []
    # vs_vals = []
    # for domain in tqdm(vs_dict):
    #     res.append(domain)
    #     z = sparse.vstack(vs_dict[domain]).mean(0)
    #     vs_vals.append(z)
        
    # mat = np.concatenate(vs_vals, 0)
    return domain_embeddings, domains

def write_split(domain, add_bos_token, num_workers, batch_size, output_dir, split, files=None, ignore_files=[]):
    curr_tokens = 0
    dataset = Domain(PROJECT_DIR / domain / domain,
                    filenames=files,
                    add_bos_token=False,
                    ignore_files=ignore_files)
    
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    
    files = []
    done = False
    with open(output_dir / f"{split}.txt", "a+") as f:
        pbar = tqdm(loader)
        
        written = False
        for fname, text in pbar:
            files.extend(fname)
            for item in text:
                curr_tokens += len(item.split())
                pbar.set_description(f"{split}, num tokens: {humanize.intword(curr_tokens)}")
                if curr_tokens > TOKEN_COUNTS[domain][f'num_{split}_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(item.split()[:TOKEN_COUNTS[domain][f'num_{split}_tokens']])
                        f.write(item + "\n")
                        written = True
                    done = True
                    break
                f.write(item + "\n")
                written = True
            if done:
                break
    return dataset.files, files, curr_tokens
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    # parser.add_argument("--add-bos-token", action='store_true')
    # parser.add_argument("--num-workers", type=int, default=0)
    # parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cluster-output-dir", type=Path)
    # parser.add_argument("--subdomains", type=Path)
    args = parser.parse_args()
        # domain = args.domain
        
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    meta = pd.read_json(PROJECT_DIR / args.domain / "metadata" / "metadata.1.jsonl", lines=True)
    populated_domains = meta.domain.value_counts()
    populated_domains  = populated_domains.loc[populated_domains > 1000]
    populated_domains = populated_domains.index
    submeta = meta.loc[meta.domain.isin(populated_domains)]
    tfidf_vecs, metadata = compute_tfidf(submeta, args.domain)
    clustering = Cluster('KMeansConstrained')
    indices = range(tfidf_vecs.shape[0])
    subdomains = clustering.cluster_vectors(vectors=tfidf_vecs, 
                                            indices=indices,
                                            n_clusters=7,
                                            size_min=50,
                                            reduce_dimensions_first=True,
                                            num_components=50,
                                            output_dir=args.cluster_output_dir)

    for idx, cluster in tqdm(subdomains['clusters'].items()):
        (output_dir / str(idx)).mkdir(exist_ok=True)
        domains = [metadata[x] for x in cluster]
        fnames = meta.loc[meta.domain.isin(domains)].filename.tolist()
        random.shuffle(fnames)
        dev_files = []
        for i in range(10000):
            dev_files.append(fnames.pop(i))
        
        random.shuffle(fnames)
        test_files = []
        for i in range(10000):
            test_files.append(fnames.pop(i))

        dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                    filenames=dev_files,
                    add_bos_token=True)
        loader = DataLoader(dataset,
                            num_workers=1,
                            batch_size=1,
                            collate_fn=collate_fn)

        with open(output_dir / str(idx) / "dev.txt", "w+") as f:
            for fname, text, _ in loader:
                for item in text:
                    f.write(item + "\n")

        dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                    filenames=test_files,
                    add_bos_token=True)
        loader = DataLoader(dataset,
                            num_workers=1,
                            batch_size=1,
                            collate_fn=collate_fn)

        with open(output_dir / str(idx) / "test.txt", "w+") as f:
            for fname, text, _ in loader:
                for item in text:
                    f.write(item + "\n")

        dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                    filenames=fnames,
                    ignore_files=dev_files + test_files,
                    add_bos_token=True)
        loader = DataLoader(dataset,
                            num_workers=16,
                            batch_size=16,
                            collate_fn=collate_fn)
        
        done = False
        pbar = tqdm(loader)
        
        with open(output_dir / str(idx) / "train.txt", "w+") as f:
            
            written = False
            curr_tokens = 0
            for fname, text, _ in pbar:
                for item in text:
                    curr_tokens += len(item.split())
                    pbar.set_description(f"train, num tokens: {humanize.intword(curr_tokens)}")
                    if curr_tokens > TOKEN_COUNTS[args.domain][f'num_train_tokens']:
                        if not written:
                            count_ = 0
                            item = " ".join(item.split()[:TOKEN_COUNTS[args.domain][f'num_train_tokens']])
                            f.write(item + "\n")
                            written = True
                        done = True
                        break
                    f.write(item + "\n")
                    written = True
                if done:
                    break
    
    idx = 7
    (output_dir / str(idx)).mkdir(exist_ok=True)
    domains = [metadata[x] for x in cluster]
    fnames = meta.loc[~meta.domain.isin(domains)].filename.tolist()
    random.shuffle(fnames)
    dev_files = []
    for i in range(10000):
        dev_files.append(fnames.pop(i))
    
    random.shuffle(fnames)
    test_files = []
    for i in range(10000):
        test_files.append(fnames.pop(i))

    dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                filenames=dev_files,
                add_bos_token=True)
    loader = DataLoader(dataset,
                        num_workers=1,
                        batch_size=1,
                        collate_fn=collate_fn)

    with open(output_dir / str(idx) / "dev.txt", "w+") as f:
        for fname, text, _ in loader:
            for item in text:
                f.write(item + "\n")

    dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                filenames=test_files,
                add_bos_token=True)
    loader = DataLoader(dataset,
                        num_workers=1,
                        batch_size=1,
                        collate_fn=collate_fn)

    with open(output_dir / str(idx) / "test.txt", "w+") as f:
        for fname, text, _ in loader:
            for item in text:
                f.write(item + "\n")

    dataset = Domain(PROJECT_DIR / args.domain / args.domain,
                filenames=fnames,
                ignore_files=dev_files + test_files,
                add_bos_token=True)
    loader = DataLoader(dataset,
                        num_workers=16,
                        batch_size=16,
                        collate_fn=collate_fn)
    
    done = False
    pbar = tqdm(loader)
    
    with open(output_dir / str(idx) / "train.txt", "w+") as f:
        
        written = False
        curr_tokens = 0
        for fname, text, _ in pbar:
            for item in text:
                curr_tokens += len(item.split())
                pbar.set_description(f"train, num tokens: {humanize.intword(curr_tokens)}")
                if curr_tokens > TOKEN_COUNTS[args.domain][f'num_train_tokens']:
                    if not written:
                        count_ = 0
                        item = " ".join(item.split()[:TOKEN_COUNTS[args.domain][f'num_train_tokens']])
                        f.write(item + "\n")
                        written = True
                    done = True
                    break
                f.write(item + "\n")
                written = True
            if done:
                break


    # subdomains, _ = load_clusters(args.subdomains)
    # labels = np.load('test.npy.labels.npy')
    # metadata = pd.read_json(PROJECT_DIR / args.domain / 'metadata' / 'metadata.1.jsonl', lines=True)
    # from
    # for key, vals in subdomains['clusters'].items():
    #     (args.output_dir / str(key)).mkdir(exist_ok=True)
    #     vals = labels[vals]
    #     for val in vals:
    #         subdomain_files = list(metadata.loc[metadata.domain == val].filename.values)
    #         train_files, train_files_to_ignore, num_train_tokens = write_split(args.domain,
    #                                                         args.add_bos_token,
    #                                                         args.num_workers,
    #                                                         args.batch_size,
    #                                                         args.output_dir / str(key),
    #                                                         "train",
    #                                                         files=subdomain_files)
    #         # dev_files, dev_files_to_ignore, num_dev_tokens = write_split(args.domain,
    #         #                                             args.add_bos_token,
    #         #                                             0,
    #         #                                             args.batch_size,
    #         #                                             args.output_dir / key,
    #         #                                             "dev",
    #         #                                             files=train_files,
    #         #                                             ignore_files=train_files_to_ignore)
    #         # _, _, num_test_tokens = write_split(args.domain,
    #         #                                     args.add_bos_token,
    #         #                                     0,
    #         #                                     args.batch_size,
    #         #                                     args.output_dir / key,
    #         #                                     "test",
    #         #                                     files=train_files,
    #         #                                     ignore_files=train_files_to_ignore + dev_files_to_ignore)


        # print("Finished successfully.")
        # print(f"Num train tokens: {humanize.intword(num_train_tokens)}")
        # # print(f"Num dev tokens: {humanize.intword(num_dev_tokens)}")
        # # print(f"Num test tokens: {humanize.intword(num_test_tokens)}")
