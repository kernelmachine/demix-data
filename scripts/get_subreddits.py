
#MIT, Apache 2.0, BSD-2, or BSD-3
from datasets import load_dataset
from collections import defaultdict
from tqdm.auto import tqdm
import gzip
import json
import os
from pathlib import Path
import jsonlines
import sys
from domain_loader.domain_loader import IterableDomain
from torch.utils.data import DataLoader
from domain_loader.constants import DATA_DIR
import humanize
batch_size = 10000

def writeall_jsonl_gz(filename, payload, dumps=None):
    with gzip.open(filename, 'wb+') as fp:
        json_writer = jsonlines.Writer(fp, dumps=dumps)
        json_writer.write_all(payload)




subreddits = {x.strip(): 1 for x in open("subreddit_list.txt", "r").readlines()}
token_count = {x: 0 for x in subreddits.keys()}
shard_count = {x: 0 for x in subreddits.keys()}
doc_count = {x: 0 for x in subreddits.keys()}
texts = defaultdict(list)


# for subreddit in tqdm(subreddits.keys()):
#     if not (DATA_DIR / subreddit / subreddit).is_dir():
#         continue
#     filenames = None
#     dataset = IterableDomain(DATA_DIR / subreddit / subreddit)

#     dataloader = DataLoader(dataset,
#                             num_workers=2,
#                             batch_size=1024)

#     pbar = tqdm(dataloader, leave=False)
#     curr_tokens = 0
#     complete = {}
#     for id, file,text, tc, _ in pbar:
#         token_count[subreddit] += sum(tc).item()
#         if not complete.get(file[-1]):
#             complete[file[-1]] = 1 
#         pbar.set_description(f"total shards: {dataset.num_files}, progress: {round(len(complete) / dataset.num_files * 100) }%, {humanize.intword(token_count[subreddit])} tokens")

# for subreddit in tqdm(subreddits.keys()):
#     if not (DATA_DIR / subreddit / subreddit).is_dir():
#         continue
#     else:
#         shard_count[subreddit] += len(list((DATA_DIR / subreddit / subreddit).glob("*.json.gz") ))

for file in ['subreddit_2019_05.json.gz', 'subreddit_2019_06.json.gz']:
    with gzip.open(file, 'rb') as f:
        pbar = tqdm(f)
        
        for line in pbar:
            line = json.loads(line)
            subreddit = line['subreddit']
            if not subreddits.get(subreddit):
                continue
            if token_count[subreddit] > 20_000_000:
                # Path(f"/private/home/suching/raw_data/subreddits_other/{subreddit}/{subreddit}/").mkdir(exist_ok=True, parents=True)
                # fname = f"/private/home/suching/raw_data/subreddits_other/{subreddit}/{subreddit}/{shard_count[subreddit]}.json.gz"
                # writeall_jsonl_gz(fname, texts[subreddit], json.dumps)
                # texts[subreddit] = []
                # shard_count[subreddit] += 1
                continue
            else:
                if line['text'] != '[removed]' and line['text'] != '[deleted]':
                    texts[subreddit].append(line)
                    token_count[subreddit] += len(line['text'].split())
                    doc_count[subreddit] += 1
                top_subreddits = [[x for x,y in sorted(token_count.items(), key=lambda x: x[1])[::-1][:10]][0]]
                s = [f"[{subreddit}] num words: {token_count[subreddit]}, num docs: {doc_count[subreddit]}, num shards: {shard_count[subreddit]}" for subreddit in top_subreddits]
                pbar.set_description(" ".join(s))
                if doc_count[subreddit] > 0 and doc_count[subreddit] % batch_size == 0:
                    Path(f"/private/home/suching/raw_data/subreddits/{subreddit}/{subreddit}/").mkdir(exist_ok=True, parents=True)
                    fname = f"/private/home/suching/raw_data/subreddits/{subreddit}/{subreddit}/{shard_count[subreddit]}.json.gz"
                    writeall_jsonl_gz(fname, texts[subreddit], json.dumps)
                    texts[subreddit] = []
                    # doc_count[subreddit] = 0
                    shard_count[subreddit] += 1

        

