import pandas as pd
from cluster.extract_features import extract_features
import argparse
from domain_loader.domain_loader import Domain, DomainVectorized
import os
from cluster.analysis import ClusterAnalysis, domain_label_map
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

DATA_DIR = "/private/home/suching/raw_data/"


label_domain_map = {y: x for x,y in domain_label_map.items()}
def read_domain(domain, files=None, sample=100):
    resolved_path = Path(DATA_DIR) / domain / domain
    if files:
        domain_files = files
    else:
        with open(Path(DATA_DIR) / domain  / "metadata" / "filenames.txt", 'r') as f:
            domain_files = []
            for x in tqdm(f.readlines()):
                fp = x.strip()
                domain_files.append(fp)

        if domain == 'reddit':
            batch_size = 1
            add_bos_token=False
        else:
            if domain != '1b':
                add_bos_token=True
            else:
                add_bos_token=False
            batch_size = 16
            sample = sample
        if sample:
            domain_files = np.random.choice(domain_files, sample)
    
    dataset = Domain(resolved_path,
                     filenames=list(domain_files),
                     add_bos_token=True, silent=True)
    dataloader = DataLoader(dataset,
                            num_workers=0,
                            batch_size=16)
    pbar = dataloader
    curr_tokens = 0
    texts = []
    metadatas = []
    for _, text, _, metadata in pbar:
        for t in text:
            if domain == 'reddit':
                docs = t.split("<|endoftext|>")
                docs = [x for x in docs if x]
                texts.extend(docs[:sample])
                break
            else:
                texts.append(t)
                metadatas.append(metadata)
        if domain == 'reddit':
            break
            
    df = pd.DataFrame({"text": texts, 'id': range(len(texts)), 'domain': label_domain_map[domain], 'metadata': metadatas})
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--path_to_file", default=None)
    parser.add_argument("--path_to_data_bins", nargs="+", default=[])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--fairseq_binary", action='store_true')

    parser.add_argument("--fname")

    parser.add_argument("--master_addr", default='127.0.0.1')
    parser.add_argument("--master_port", default='29500')
    args = parser.parse_args()
    print("reading metadata...")
    # df = pd.read_json("/private/home/suching/raw_data/openwebtext/metadata/metadata_full_fnames.jsonl", lines=True)

    # domain_counts = df.domain.value_counts()
    # domains = domain_counts.loc[domain_counts > 50].index
    # tqdm.pandas()
    # z = df.loc[df.domain.isin(domains)].groupby('domain').progress_apply(lambda x: x.sample(n=min(50, x.shape[0]))).reset_index(drop=True)
    # z['filename'] = z.filename.apply(lambda x: Path(DATA_DIR) / "openwebtext" / "openwebtext" / x)
    # print("reading files...")
    # texts = z.groupby('domain').progress_apply(lambda x: read_domain('openwebtext', files=x.filename.tolist()))
    # texts = texts.drop(['domain'], axis=1).reset_index()
    # texts['id'] = range(len(texts))
    extract_features(domain=args.domain,
                 path_to_file=args.path_to_file,
                 path_to_data_bins=args.path_to_data_bins,
                 model=args.model,
                 batch_size=args.batch_size,
                 num_gpus=args.num_gpus,
                 fname=args.fname,
                 master_addr=args.master_addr,
                 master_port=args.master_port,
                 sample=args.sample,
                 notebook=False)