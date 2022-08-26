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


# {
#     "Assembly": [".asm"],
#     "Batchfile": [".bat", ".cmd"],
#     "C": [".c", ".h"],
#     "C#": [".cs"],
#     "C++": [".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H"],
#     "CMake": [".cmake"],
#     "CSS": [".css"],
#     "Dockerfile": [".dockerfile", "Dockerfile"],
#     "FORTRAN": ['.f90', '.f', '.f03', '.f08', '.f77', '.f95', '.for', '.fpp'],
#     "GO": [".go"],
#     "Haskell": [".hs"],
#     "HTML":[".html"],
#     "Java": [".java"],
#     "JavaScript": [".js"],
#     "Julia": [".jl"],
#     "Lua": [".lua"],
#     "Makefile": ["Makefile"],
#     "Markdown": [".md", ".markdown"],
#     "PHP": [".php", ".php3", ".php4", ".php5", ".phps", ".phpt"],
#     "Perl": [".pl", ".pm", ".pod", ".perl"],
#     "PowerShell": ['.ps1', '.psd1', '.psm1'],
#     "Python": [".py"],
#     "Ruby": [".rb"],
#     "Rust": [".rs"],
#     "SQL": [".sql"],
#     "Scala": [".scala"],
#     "Shell": [".sh", ".bash", ".command", ".zsh"],
#     "TypeScript": [".ts", ".tsx"],
#     "TeX": [".tex"],
#     "Visual Basic": [".vb"]
# }


language = sys.argv[1]

ds = load_dataset("lvwerra/github-code",
                 streaming=True,
                 split="train",
                 licenses=["mit", "apache-2.0", "bsd-2-clause", 'bsd-3-clause'],
                 languages=[language])

license_map = {"mit": 1, "apache-2.0": 1, "bsd-2-clause": 1, 'bsd-3-clause': 1}
counts = defaultdict(int)
licenses = []
iterable = iter(ds)

data = defaultdict(list)
pbar = tqdm(iterable)
batch_size = 10000



def writeall_jsonl_gz(filename, payload, dumps=None):
    with gzip.open(filename, 'wb+') as fp:
        json_writer = jsonlines.Writer(fp, dumps=dumps)
        json_writer.write_all(payload)

counter = 0
for element in pbar:
    if counts[language] > 10_000_000_000:
        break
    else:
        counts[language] += len(element['code'].split())
        data[language].append(element)
        num_data = len(data[language])  
        s = [f"[{x}] num words: {y}, num docs: {num_data}, num shards: {counter}" for x,y in counts.items()]
        pbar.set_description(" ".join(s)) 
        if num_data > 0 and num_data % batch_size == 0:
            Path(f"/private/home/suching/raw_data/programming_languages_redo/{language}/").mkdir(exist_ok=True)
            fname = f"/private/home/suching/raw_data/programming_languages_redo/{language}/{counter}.json.gz"
            writeall_jsonl_gz(fname, data[language], json.dumps)
            data[language] = []
            counter += 1

print(f"Completed {language}, num words: {counts[language]}, num shards: {counter}")