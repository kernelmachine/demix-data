# Download Instructions for DEMix Data

Here we provide instructions to download data used in the DEMix paper. Note that downloading most datasets involve getting approval from dataset hosters.

### 1B Words

Download 1B words corpus from here: https://opensource.google/projects/lm-benchmark

### Legal

Create an account and download data here https://case.law/

### S2ORC (e.g., Med, CS)

Follow instructions here to download papers: https://github.com/allenai/s2orc

When papers are downloaded, you can extract papers using the scripts in `domains/s2orc/extract_papers.py`.

### Openwebtext

Download Openwebtext from here https://skylion007.github.io/OpenWebTextCorpus/.

Use the script at `domains/openwebtext/unpack_openwebtext.py` to unpack the data.

### RealNews

Download the dataset from here: https://docs.google.com/forms/d/1LMAUeUtHNPXO9koyAIlDpvyKsLSYlrBj3rYhC30a7Ak/viewform?edit_requested=true

### Reviews

Download the raw review data from here: http://deepyeti.ucsd.edu/jianmo/amazon/index.html

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

Sign up for the [Twitter Academic API](https://developer.twitter.com/en/products/twitter-api/academic-research), and download tweets in a jsonl format.


### Breaking News

Use `domain/scripts/fetch_articles.py` to crawl breaking news articles. We use the URLs associated with `high factuality` in `https://github.com/ramybaly/News-Media-Reliability/blob/master/data/acl2020/corpus.tsv`.

```bash
python -m domain.scripts.fetch_articles --num-articles-per-source 100 --path-to-output news.jsonl
```


### Yelp Reviews

Download dataset here https://www.yelp.com/dataset
