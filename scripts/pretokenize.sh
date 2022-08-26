DIR=$1
PATH_TO_GPT2_BPE=$2
for SPLIT in train dev test; do \
        python -m scripts.multiprocessing_bpe_encoder \
                --encoder-json ${PATH_TO_GPT2_BPE}/encoder.json \
                --vocab-bpe ${PATH_TO_GPT2_BPE}/vocab.bpe \
                --inputs  ${DIR}/${SPLIT}.txt \
                --outputs ${DIR}/${SPLIT}.txt.bpe \
                --keep-empty \
                --workers 60;
done
