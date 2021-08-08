DIR=$1
for SPLIT in train dev test; do \
        python -m scripts.multiprocessing_bpe_encoder \
                --encoder-json ${DATA_DIR}/gpt2_bpe/encoder.json \
                --vocab-bpe ${DATA_DIR}/gpt2_bpe/vocab.bpe \
                --inputs  ${DIR}/${SPLIT}.txt \
                --outputs ${DIR}/${SPLIT}.txt.bpe \
                --keep-empty \
                --workers 60;
done
