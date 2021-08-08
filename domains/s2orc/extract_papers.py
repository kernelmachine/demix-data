from glob import glob
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import gzip
import random
import json
from joblib import Parallel, delayed
pd.options.mode.chained_assignment = None  # default='warn'
import argparse

def get_papers(texts_dir, file, paper_ids_to_keep, ix):
    with gzip.open(file, 'rb') as f, open(texts_dir / 'metadata.jsonl', 'a+') as g:
        papers = []
        pbar = tqdm(f)
        for line in pbar:
            if not line:
                continue
            z = json.loads(line)
            if paper_ids_to_keep.get(z['paper_id']):
                pbar.set_description(f"papers to extract: {len(paper_ids_to_keep)}, written {ix} subsets")
                _ = paper_ids_to_keep.pop(z['paper_id'])
                z['filename'] = str(Path(f"subset_{ix}") / (str(random.getrandbits(128)) + ".txt"))
                text = " ".join([paper['text'] for paper in z['body_text']])
                if text:
                    papers.append((z['filename'], text))
                json.dump(z, g)
                g.write("\n")
            if len(papers) > 512:
                (texts_dir / f"subset_{ix}").mkdir(parents=True, exist_ok=True)
                for fname, paper in papers:
                    with open(texts_dir / fname, 'w') as h:
                        h.write(paper)
                pbar.set_description(f"papers to extract: {len(paper_ids_to_keep)}, written {ix} subsets")
                ix += 1
                papers = []

    return ix, paper_ids_to_keep


def get_metadata(file, name):
    with gzip.open(file, 'rb') as f, open(f"metadata/{name}.jsonl", "a+") as g:
        for line in tqdm(f, leave=False):
            if not line:
                continue
            z = json.loads(line)
            if not z.get('mag_field_of_study'):
                continue
            if set(z['mag_field_of_study']) & fields_of_study:
                json.dump(z, g)
                f.write('\n')
    return metadata



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields-of-study", type=str, nargs='+')
    args = parser.parse_args()

    fields_of_study = args.fields_of_study

    metadata_files = list(Path('20200705v1/full/metadata/').rglob('*'))
    pdf_parses = list(Path('20200705v1/full/pdf_parses/').rglob('*'))



    for name in fields_of_study:
        texts_dir = Path(f'{name}/')
        texts_dir.mkdir(exist_ok=True)
        paper_ids_to_keep = {}
        with open(f"metadata/{name}.jsonl", "r") as f:
            for line in tqdm(f):
                line = json.loads(line)
                paper_ids_to_keep[line['paper_id']] = 1
        ix, i = 0, 0
        files_pbar = tqdm(total=len(pdf_parses))
        while paper_ids_to_keep and i < len(pdf_parses):
            ix, paper_ids_to_keep = get_papers(texts_dir, pdf_parses[i], paper_ids_to_keep, ix)
            i += 1
            files_pbar.update(1)
        if i >= len(pdf_parses):
            print(f"reached end of input")
        elif not paper_ids_to_keep:
            print(f"finished extracting all papers")
