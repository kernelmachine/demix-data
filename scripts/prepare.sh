DOMAIN=$1
NUM_WORKERS=$2
BATCH_SIZE=$3
OUTPUT_DIR=$4
#python -m loader.make_splits --domain $DOMAIN  --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --output-dir $DOMAIN/splits-big/
bash scripts/pretokenize.sh $DOMAIN
bash scripts/preprocess.sh $DOMAIN $OUTPUT_DIR
