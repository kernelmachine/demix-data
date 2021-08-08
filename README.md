# DEMix Data

This repository contains data utilities for DEMix Layers: Disentangling Domains for Modular Language Modeling

We assume a sharded dataset structure across domains, where the dataset is split among many folders, and each folder contains many files, each containing their own document. We found this format to be particularly amenable to PyTorch dataloading. This follows the Openwebtext dataset format.

We additionally assume the existence of a `metadata/` folder, which contains `filenames.txt`, an index of the paths to all files in your dataset, and a `metadata.jsonl`, a json-lines file which contains per-document metadata.

Please see the `examples/` directory for an example of this dataset structure. The scripts referenced below download and format each domain's dataset into this structure, but you can write your own scripts to do this.

## Installation

First, set your `PROJECT_DIR` to the root directory where you will be housing the domain directories.

```bash
export PROJECT_DIR=$(pwd)
```

```bash
export DATA_PATH=/private/home/suching/raw_data/
```

## Download data

We provide an example data input file in `example_domains/`
```bash
export DOMAIN=imdb
```


## Shard Data

```bash
python -m domain_loader.build_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 16 --text-field text
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
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $PROJECT_DIR/$DOMAIN/splits
```


## Build fairseq data-bin


Download the gpt2 vocabulary:

```bash
mkdir ${PROJECT_DIR}/gpt2_bpe
curl -Lo ${PROJECT_DIR}/gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
curl -Lo ${PROJECT_DIR}/gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
curl -Lo ${PROJECT_DIR}/gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
```

```bash
bash scripts/pretokenize.sh ${PROJECT_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${PROJECT_DIR}/$DOMAIN/splits $DOMAIN ${PROJECT_DIR}/data-bin/
```

These scripts will output a `data-bin` files in `${PROJECT_DIR}/data-bin/`, which you can train on with fairseq LMs.



## Building multi-domain datasets


Building a multi-domain dataset follows the same procedure above, except you just add multiple domains in the same data-bin folder (i.e., `${PROJECT_DIR}/data-bin/`).

You can apply the same process to the `ag-news` domain:

```bash
export DOMAIN=ag_news
python -m domain_loader.build_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 16 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "ag_news" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $PROJECT_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${PROJECT_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${PROJECT_DIR}/$DOMAIN/splits $DOMAIN ${PROJECT_DIR}/data-bin/
```
