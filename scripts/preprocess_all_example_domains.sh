export DOMAIN=ag_news
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "ag_news" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/


export DOMAIN=imdb
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "imdb" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/


export DOMAIN=1b_test
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "sciie" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/


export DOMAIN=hp-news
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "hp-news" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/


export DOMAIN=chemprot
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "chemprot" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/


export DOMAIN=rct
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "rct" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 16 --batch-size 16 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/

export DOMAIN=citation_intent
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "citation_intent" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/

export DOMAIN=amazon
python -m domain_loader.shard_dataset --domain $DOMAIN --input-file example_domains/$DOMAIN/$DOMAIN.jsonl --batch-size 512 --text-field text
python -m domain_loader.scan_filenames --domain $DOMAIN
python -m domain_loader.count_words --domain $DOMAIN
## set token counts for "amazon" in domain_loader/constants.py
python -m domain_loader.make_splits --domain $DOMAIN --num-workers 0 --batch-size 1 --output-dir $DATA_DIR/$DOMAIN/splits
bash scripts/pretokenize.sh ${DATA_DIR}/$DOMAIN/splits
bash scripts/preprocess.sh ${DATA_DIR}/$DOMAIN/splits $DOMAIN ${DATA_DIR}/data-bin/
