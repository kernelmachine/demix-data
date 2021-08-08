INPUT_DIR=$1
DOMAIN=$2
OUTPUT_DIR=$3

fairseq-preprocess \
            --only-source \
			--srcdict ${DATA_DIR}/gpt2_bpe/dict.txt \
			--trainpref ${INPUT_DIR}/train.txt.bpe \
			--validpref ${INPUT_DIR}/dev.txt.bpe \
			--testpref ${INPUT_DIR}/test.txt.bpe \
			--destdir ${OUTPUT_DIR}/${DOMAIN} \
            --workers 60;
mv ${OUTPUT_DIR}/${DOMAIN}/valid.bin ${OUTPUT_DIR}/${DOMAIN}/valid_${DOMAIN}.bin
mv ${OUTPUT_DIR}/${DOMAIN}/valid.idx ${OUTPUT_DIR}/${DOMAIN}/valid_${DOMAIN}.idx
mv ${OUTPUT_DIR}/${DOMAIN}/test.bin ${OUTPUT_DIR}/${DOMAIN}/test_${DOMAIN}.bin
mv ${OUTPUT_DIR}/${DOMAIN}/test.idx ${OUTPUT_DIR}/${DOMAIN}/test_${DOMAIN}.idx
cp ${DATA_DIR}/gpt2_bpe/dict.txt ${OUTPUT_DIR}/dict.txt
