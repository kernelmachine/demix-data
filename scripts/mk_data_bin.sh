DATA_DIR=$1
DOMAIN=$2
OUTPUT_DIR=$3

bash scripts/pretokenize.sh ${DATA_DIR}/${DOMAIN}/splits/ /private/home/suching/raw_data/gpt2_bpe/
bash scripts/preprocess.sh ${DATA_DIR}/${DOMAIN}/splits/ ${DOMAIN} ${OUTPUT_DIR} /private/home/suching/raw_data/gpt2_bpe/

