from joblib import Parallel, delayed
from pathlib import Path
import gzip
import os
from tqdm.auto import tqdm

def write_docs_to_directory(chunk, docs):
    acc_docs = []
    for ix, doc in tqdm(enumerate(docs), total=len(docs)):
        if ix % 1024 == 0:
            Path(f'/private/home/suching/raw_data/reddit_flat/reddit_flat/{chunk}/subset_{ix}').mkdir(parents=True,exist_ok=True)
            for i, d in enumerate(acc_docs):
                with open(f'/private/home/suching/raw_data/reddit_flat/reddit_flat/{chunk}/subset_{ix}/{i}.txt', 'w+') as f:
                    f.write(d)
            acc_docs = []
        else:
            acc_docs.append(doc)

def write_chunk(file):
    chunk = file.split('/')[-1].split('.')[0]
    with gzip.open(file, 'rb') as f:
        z = f.read().decode('utf-8')
    docs = z.split('<|endoftext|>')
    if Path(f"/private/home/suching/raw_data/reddit_flat/reddit_flat/{chunk}/").exists():
        return
    else:
        write_docs_to_directory(chunk, docs)



if __name__ == '__main__':
    files = []
    root = "/private/home/suching/raw_data/reddit/reddit/"
    for dir_ in os.listdir(root):
        for file in os.listdir(os.path.join(root, dir_)):
            full_path = os.path.join(root, dir_, file)
            files.append(full_path)

    Parallel(n_jobs=64)(delayed(write_chunk)(file) for file in tqdm(files))
        