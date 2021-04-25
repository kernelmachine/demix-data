import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
from collections import defaultdict

from typing import List
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import pandas as pd
from domain_loader.domain_loader import Domain
from torch.utils.data import DataLoader
from domain_loader.constants import PROJECT_DIR
sns.set(context="paper", style="white", font_scale=1.4) 

def load_data(data_path: str, sample: int=None) -> List[str]:
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}", disable=sample is None) as f:
        for line in f:
            if sample:
                if len(examples) > sample:
                    break
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {"text": line}
                text = example['text']
                if sample:
                    if np.random.binomial(1, 0.5):
                        examples.append(text)
                else:
                    examples.append(text)
    if sample:
        examples = np.random.choice(examples, size=sample)
    return examples

def load_vocab(loader):
    count_vectorizer = CountVectorizer(min_df=3, max_features=10000, stop_words="english", ngram_range=(2,2))
    pbar = tqdm(text)
    pbar.set_description(file)
    count_vectorizer.fit(pbar)
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_plot_file", help="path to save heatmap", required=True)
    parser.add_argument("--output_data_file", help="path to save heatmap data", required=True)
    parser.add_argument("--sample", type=int, help="sample documents", required=False)
    args = parser.parse_args()

    domain = "openwebtext"
   
    dataset = Domain(PROJECT_DIR / domain / domain,
                    filenames=None,
                    add_bos_token=False,
                    ignore_files=[],
                    metadata_columns=["domain"],
                    metadata_file=PROJECT_DIR / domain / "metadata" / "metadata.1.jsonl",
                    sample_by_metadata={"metadata_column": "domain", "sample_size": 10},
                    domain=populated_domains)
    
    dataloader = DataLoader(dataset,
                            num_workers=1,
                            batch_size=1)
                            
    
    pbar = tqdm(dataloader)
    curr_tokens = 0
    vocabs = {}
    ix = 0
    for _, text, metadata in pbar:
        count_vectorizer = CountVectorizer(max_features=10000, stop_words="english", ngram_range=(2,2))
        count_vectorizer.fit(text)
        vocabs[metadata[0][0]] = set(count_vectorizer.vocabulary_.keys())


    file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    
    overlaps = {}
    for x, y in tqdm(file_pairs):
        intersection = vocabs[x] & vocabs[y]
        union = (vocabs[x] | vocabs[y])
        overlaps[x + "_" + y] = len(intersection) / len(union)
    
    data = []

    z = {}
    for key in tqdm(overlaps.keys()):
        file_1, file_2 = key.split('_')
        if not z.get(file_1):
            z[file_1] = {}
        z[file_1][file_2] = overlaps[key]
        if not z.get(file_2):
            z[file_2] = {}
        z[file_2][file_1] = overlaps[key]

    labels = list(vocabs.keys())

    for ix, key in tqdm(enumerate(z)):
        items = []
        for subkey in labels:
            if not z[key].get(subkey):
                items.append(1.0)
            else:
                items.append(z[key][subkey])
        data.append(items)
    
    data = np.array(data) * 100
    if args.output_data_file:
        print('saving data...')
        np.save(args.output_data_file + ".labels.npy", np.array(labels))
        np.save(args.output_data_file, data)
    # print('generating fig...')
    # fig, ax = plt.subplots(1,1,figsize=(8,8))
    # sns.heatmap(data, cmap="Blues",  cbar=True,  ax=ax)
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # print('saving fig...')
    # plt.savefig(args.output_plot_file, dpi=300)
