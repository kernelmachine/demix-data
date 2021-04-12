import json
from typing import Iterable, List, TypeVar
from tqdm.auto import tqdm
from pathlib import Path
import random


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
    texts_dir = Path('legal/')
    
    files = list(Path(".").glob('*/data/data.jsonl'))
    print(files)
    for file in files:  
        with open(file, 'r') as f, open('metadata/metadata.jsonl', 'a+') as g:
            for ix, batch in tqdm(enumerate(batchify(f, batch_size=512))):
                batch = [json.loads(x) for x in batch]
                for x in batch:
                    x['filenames'] = []
                    for y in x['casebody']['data']['opinions']:
                        fname = random.getrandbits(128)
                        x['filenames'].append(str(Path(f"subset_{ix}") / (str(random.getrandbits(128)) + ".txt")))

                text = [zip(x['filenames'], [x['text'] for x in x['casebody']['data']['opinions']]) for x in batch]
                for line in batch:
                    json.dump(line, g)
                    g.write("\n")
                (texts_dir / f"subset_{ix}").mkdir(parents=True, exist_ok=True)
                for x in text:
                    for fname, line in x: 
                        with open(texts_dir / fname, 'w') as h:
                            h.write(line)
