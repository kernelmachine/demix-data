# DEMix Data

Data utilities for DEMix Layers: Disentangling Domains for Modular Language Modeling


## Installation

```bash
export DATA_PATH=/private/home/suching/raw_data/
```

## Download datasets

Here we provide instructions to download data. Most datasets involve getting approval from dataset hosters.

### 1B Words

Download 1B words corpus from here: https://opensource.google/projects/lm-benchmark

### Legal

Create an account and download data here https://case.law/


Use the script at `domains/legal/split_legal.py` to split the resulting jsonl files into separate files.


### S2ORC (e.g., Med, CS)

Follow instructions here to download papers: https://github.com/allenai/s2orc

When papers are downloaded, you can extract papers using the scripts in `domains/s2orc/extract_papers.py`.

### Openwebtext

Download Openwebtext from here https://skylion007.github.io/OpenWebTextCorpus/.

Use the script at `domains/openwebtext/unpack_openwebtext.py` to unpack the data.

### RealNews

Download the dataset from here: https://docs.google.com/forms/d/1LMAUeUtHNPXO9koyAIlDpvyKsLSYlrBj3rYhC30a7Ak/viewform?edit_requested=true

Use the script at `domains/realnews/split_realnews.py` to split the resulting jsonl file into separate files.

### Reviews

Download the raw review data from here: http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Use the script at `domains/reviews/split_reviews.py` to split the resulting jsonl file into separate files.

### Gutenberg

Follow instructions here to download the data: https://github.com/aparrish/gutenberg-dammit

### Github

Download data here: https://console.cloud.google.com/marketplace/product/github/github-repos, under the `contents` table.

### ACL Papers

Download data here: https://allenai.org/data/qasper

### Legal contracts

Download data here: https://www.atticusprojectai.org/cuad

### CORD-19

Download dataset here: https://www.semanticscholar.org/cord19/download

### Tweets

Sign up for the [Twitter Academic API](https://developer.twitter.com/en/products/twitter-api/academic-research), and download tweets in a jsonl format. Then split into files using a process similar to `domains/realnews/split_realnews.py`.

### Breaking News

Use `domain/scripts/fetch_articles.py` to crawl breaking news articles. We use the URLs associated with `high factuality` in `https://github.com/ramybaly/News-Media-Reliability/blob/master/data/acl2020/corpus.tsv`.

```bash
python -m domain.scripts.fetch_articles --num-articles-per-source 100 --path-to-output news.jsonl
```

Then split into files using a process similar to `domains/realnews/split_realnews.py`.


### Yelp Reviews

Download dataset here https://www.yelp.com/dataset

Then split into files using a process similar to `domains/realnews/split_realnews.py`.

## Build fairseq data-bin

```bash
python -m domains.s2orc.extract_papers
```
```bash
export DOMAIN=med
export NUM_WORKERS=16
export BATCH_SIZE=16
export DATA_BIN_DIR=${DATA_PATH}/data-bin/
python -m domain_loader.make_splits --domain $DOMAIN  --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --output-dir ${DATA_PATH}/${DOMAIN}/splits-big/
bash scripts/pretokenize.sh $DOMAIN
bash scripts/preprocess.sh $DOMAIN $DATA_BIN_DIR
```

These scripts will output a `data-bin` files in $DATA_BIN_DIR, which you can train on with fairseq LMs.
