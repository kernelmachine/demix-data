DOMAIN=$1
NUM_WORKERS=$2
BATCH_SIZE=$3
python -m loader.make_splits --domain $DOMAIN  --num-workers $NUM_WORKERS --batch-size $BATCH_SIZE --output-dir $DOMAIN/splits-big/
bash pretokenize.sh $DOMAIN
bash preprocess.sh $DOMAIN 
