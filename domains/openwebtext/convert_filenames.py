from tqdm.auto import tqdm
import pandas as pd
import json
from pathlib import Path
from  copy import copy

if __name__ == '__main__':
    
    fs = {}
    for file in tqdm(Path("openwebtext").glob("*/*")):
        fs[file.stem] = str(Path(str(file.parents[0]).split('/')[1]) / file.name)
    with open('metadata/metadata.jsonl', 'r') as f, open('metadata/metadata.1.jsonl', 'w+') as g:
        for line in tqdm(f):
            z = json.loads(line)
            z['stem'] = copy(z['filename'])
            z['filename'] = fs[Path(z['filename']).stem]
            json.dump(z, g)
            g.write('\n')