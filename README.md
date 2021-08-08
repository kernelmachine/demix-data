# Multidomain Language Modeling Data Utilities

This repository contains data utilities for "DEMix Layers: Disentangling Domains for Modular Language Modeling" (Gururangan et. al, 2021).

Generally, this code can be used to build data binaries in a format compatible with Fairseq for language modeling. We follow the process necessary to reproduce results in the DEMix paper.


## General Overview

In the DEMix paper, we assume a sharded dataset structure across domains, where the dataset is split among many folders, and each folder contains many files, each containing a single document. We found this format to be particularly amenable to efficient PyTorch dataloading, and this follows the Openwebtext dataset format.

The processing steps below generally build the following files:
    * A `shards/` folder, which contains a sharded version of the dataset for efficient Pytorch data loading.
    * A `data-bin/` folder, which contains data binaries for training and evaluation of language models in Fairseq
    * A `metadata/` folder, which contains `filenames.txt`, an index of the paths to all files in your dataset, and a `metadata.jsonl`, a json-lines file which contains per-document metadata. The former is used for faster data loading, and the later is used for finer-grained filtering of documents based on certain metadata.

In this tutorial, we use the example datasets in the `example_domains/` directory to build these necessary folders and files. You can use the same process on any data of any size, provided that the original input data is in a `.jsonl` format. It's easy to convert a raw text file into a `.jsonl` format (see [below](#converting_to_jsonl)).

## Installation

```bash
conda env create --name demix -f environment.yml
conda activate demix
```

First, set your `DATA_DIR` to the root directory where you will be housing the domain directories.

```bash
export DATA_DIR=$(pwd)/example_domains
```

## Download data

We provide an example data input files in the directories in `example_domains/`.

We will first preprocess the `imdb` domain.

```bash
export DOMAIN=imdb
```

Check this [file](DOWNLOAD_DATA.md) for more information on how to download the data used in the DEMix paper.

### Converting to jsonl from raw text or CSV

Sometimes the data you want to preprocess is not in jsonl to begin with. If each line in a file `raw_text.txt` is a new document, it's easy to convert it to jsonl with `jq`:

```bash
jq -R '{"text": .}' raw_text.txt
```

If you've got a CSV instead, you can use the script in `scripts/csv2json.jq` to convert this to jsonl:

```bash
jq -R -s -f scripts/csv2json.jq text.csv | jq -c .[]
```

Of course, you might find it easier to do the conversion in python, based on how large your dataset is you'd like to convert.


## Shard Data

```bash
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 16 --text-field text
```


## Build metadata/filenames.txt

To make data loading faster, we first gather a list of filenames in a separate file `${DOMAIN}/metadata/filenames.txt`. To build this file, use `domain_loader/build_filenames.py`.

```bash
python -m domain_loader.scan_filenames --domain $DOMAIN
```

## Count words

```bash
python -m domain_loader.count_words --domain $DOMAIN
```


## Split data into train, dev, and test files


We set the total number of tokens for train, dev, and test splits in `domain_loader/constants.py`.

```bash
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
```


## Build fairseq data-bin


Download the gpt2 vocabulary:

```bash
mkdir ${DATA_DIR}/gpt2_bpe
curl -Lo ${DATA_DIR}/gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
curl -Lo ${DATA_DIR}/gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
curl -Lo ${DATA_DIR}/gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
```

```bash
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/
```

These scripts will output a `data-bin` files in `${DATA_DIR}/data-bin/`, which you can train on with fairseq LMs.



## Building multi-domain datasets


Building a multi-domain dataset follows the same procedure above, except you just add multiple domains in the same data-bin folder (i.e., `${DATA_DIR}/data-bin/`).

You can apply the same process to the `ag-news` domain:

```bash
export DOMAIN=ag_news
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 16 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "ag_news" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/
```
