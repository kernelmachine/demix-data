import pandas as pd
import re
import os
import json
from tqdm.auto import tqdm
import sys


x = sys.argv[1]
y= sys.argv[2]

dig1 = re.findall(r'\d+', x)[0]
dig2 = re.findall(r'\d+', y)[0]
assert dig1 == dig2
print("reading...")
df = pd.read_json("20200705v1/full/metadata/" + x, lines=True)
df1 = pd.read_json("20200705v1/full/pdf_parses/" + y, lines=True)
df['id'] = df['paper_id']
df = df.drop(['paper_id'], axis=1)
print("merging...")
master = df.merge(df1)
for name, group in tqdm(master.explode('domain').groupby('domain')):
    if not os.path.isdir(f"/private/home/suching/raw_data/s2orc/{name}/"):
        os.mkdir(f"/private/home/suching/raw_data/s2orc/{name}/")
    group.to_json(f"/private/home/suching/raw_data/s2orc/{name}/{dig1}.json.gz", lines=True, orient='records', compression='gzip')



