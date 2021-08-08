# DEMix Data

Data utilities for DEMix Layers: Disentangling Domains for Modular Language Modeling


## Installation

```bash
export DATA_PATH=/private/home/suching/raw_data/
```

### Build fairseq data-bin

```bash
export DOMAIN=med
export NUM_WORKERS=16
export BATCH_SIZE=16
export DATA_BIN_DIR=${DATA_PATH}/data-bin/
python -m loader.make_splits --domain $DOMAIN  --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --output-dir ${DATA_PATH}/${DOMAIN}/splits-big/
bash scripts/pretokenize.sh $DOMAIN
bash scripts/preprocess.sh $DOMAIN $DATA_BIN_DIR
```
