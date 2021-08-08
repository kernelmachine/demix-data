import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
from collections import defaultdict
import humanize
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
from domain_loader.constants import DATA_DIR
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

def load_text(domain, add_bos_token=False, num_workers=1, batch_size=1, num_expected_tokens=None, num_expected_docs=None):
    with open(DATA_DIR / domain  / "splits-final" / "train_files.txt", 'r') as f:
        files = [x.strip() for x in tqdm(f.readlines())]
        np.random.shuffle(files)

    dataset = Domain(DATA_DIR / domain / domain,
                     filenames=files if domain not in ['1b', 'reddit'] else None,
                     add_bos_token=add_bos_token,
                     track_token_count=True)

    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    pbar = tqdm(loader)
    texts = []

    curr_tokens = 0
    curr_docs = 0
    written = False
    for _, text ,token_count, _ in pbar:
        s = f"{domain}, num tokens: {humanize.intword(curr_tokens)}, num docs: {humanize.intword(curr_docs)}"
        pbar.set_description(s)
        if (num_expected_docs and curr_docs > num_expected_docs):
            texts = texts[:num_expected_docs]
            break
        if domain in ['1b', 'reddit']:
            tt = [t.split("<|endoftext|>") for t in text]
            text = [y for x in tt for y in x]
            token_count = [len(x.split()) for x in text]
        if (num_expected_tokens and curr_tokens > num_expected_tokens):
            if not written:
                text = " ".join(text)[:num_expected_tokens]
                texts.extend(text)
            else:
                texts = "\n".join(texts)[:num_expected_tokens]
                texts = texts.split('\n')
                curr_tokens = num_expected_tokens
            break
        else:
            texts.extend(text)
            curr_tokens += sum(token_count)
            curr_docs += len(text)
        written = True
    return texts, curr_tokens, curr_docs

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
    parser.add_argument("--sample", type=int, help="sample tokens", required=False)
    args = parser.parse_args()
    vocabs = {}
    for domain in ['1b', 'cs', 'legal', 'med', 'openwebtext', 'realnews', 'reviews', 'reddit']:
        texts, curr_tokens, curr_docs = load_text(domain,
                                            add_bos_token=True if domain not in ['1b', 'reddit'] else False,
                                            num_workers=16 if domain not in ['1b', 'reddit'] else 1,
                                            batch_size=16 if domain not in ['1b', 'reddit'] else 1,
                                            num_expected_tokens=args.sample,
                                            num_expected_docs=None)
        count_vectorizer = CountVectorizer(stop_words="english", min_df=3, ngram_range=(2,2))
        count_vectorizer.fit(tqdm(texts))
        vocabs[domain] = set(count_vectorizer.vocabulary_.keys())


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
        np.save(args.output_data_file, data)

    print('generating fig...')
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    sns.heatmap(data, cmap="Blues",  cbar=True, annot=True,  ax=ax)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    print('saving fig...')
    plt.savefig(args.output_plot_file, dpi=300)
