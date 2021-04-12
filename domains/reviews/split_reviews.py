import json
from typing import Iterable, List, TypeVar
from tqdm.auto import tqdm
from pathlib import Path
import random
import gzip


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


if __name__ == '__main__':
    texts_dir = Path('reviews/')
    with gzip.open('All_Amazon_Review.json.gz', 'rb') as f, open('metadata/metadata.jsonl', 'w+') as g:
        for ix, batch in tqdm(enumerate(batchify(f, batch_size=512))):
            batch = [json.loads(x) for x in batch]
            for x in batch:
                fname = random.getrandbits(128)
                x['filename'] = str(Path(f"subset_{ix}") / (str(random.getrandbits(128)) + ".txt"))
            text = [(x['filename'], x.pop('reviewText')) for x in batch if x.get('reviewText')]

            for line in batch:
                json.dump(line, g)
                g.write("\n")
            (texts_dir / f"subset_{ix}").mkdir(parents=True, exist_ok=True)
            for fname, line in text:
                with open(texts_dir / fname, 'w') as h:
                    h.write(line)
