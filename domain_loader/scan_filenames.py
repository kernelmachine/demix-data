import os
from typing import Optional, List, Tuple

from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path

from domain_loader.constants import PROJECT_DIR
from tqdm.auto import tqdm
from domain_loader.domain_loader import Domain
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    domain = args.domain

    dataset = Domain(PROJECT_DIR / domain / "subsets")

    if domain in ["1b", "reddit"]:
        num_workers = 1
        batch_size = 1
    else:
        num_workers = args.num_workers
        batch_size = args.batch_size
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size)

    pbar = tqdm(dataloader)
    curr_files = 0
    (PROJECT_DIR / domain / "metadata" ).mkdir(exist_ok=True)
    with open(PROJECT_DIR / domain / "metadata" / "filenames.txt", "w+") as f:
        for fname, _, _, _ in pbar:
            for fn in fname:
                f.write(fn + "\n")
                curr_files += 1
    print(f"Number of files in {str(PROJECT_DIR / domain / domain)}: {curr_files}")
