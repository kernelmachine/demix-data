import json
from typing import Iterable, List, TypeVar
from tqdm.auto import tqdm
from pathlib import Path
import random
import gzip
import os
import argparse

T = TypeVar('T')


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def build(texts_dir, input_fh, metadata_fh, batch_size=512, text_field='text'):
    for ix, batch in tqdm(enumerate(batchify(f, batch_size=batch_size))):
        batch = [json.loads(x) for x in batch]
        for x in batch:
            fname = random.getrandbits(128)
            x['filename'] = str(Path("subsets") / f"subset_{ix}" / (str(random.getrandbits(128)) + ".txt"))
        text = [(x['filename'], x.pop(text_field)) for x in batch if x.get(text_field)]
        for line in batch:
            json.dump(line, g)
            g.write("\n")
        subset_dir = (texts_dir / "subsets" / f"subset_{ix}")
        subset_dir.mkdir(parents=True, exist_ok=True)
        for fname, line in text:
            with open(texts_dir / fname, 'w+') as h:
                h.write(line)

if __name__ == '__main__':
    project_dir = Path(os.environ['PROJECT_DIR'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--text-field", type=str, default='text')


    args = parser.parse_args()

    texts_dir = project_dir / args.domain

    (texts_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    (texts_dir / 'subsets').mkdir(parents=True, exist_ok=True)

    if args.input_file.endswith(".gz"):
        with gzip.open(args.input_file, 'rb') as f, open(texts_dir / 'metadata' / 'metadata.jsonl', 'w+') as g:
            build(texts_dir, f, g, args.batch_size, args.text_field)
    else:
        with open(args.input_file, 'r') as f, open(texts_dir/ 'metadata'/ 'metadata.jsonl', 'w+') as g:
            build(texts_dir, f, g, args.batch_size, args.text_field)
