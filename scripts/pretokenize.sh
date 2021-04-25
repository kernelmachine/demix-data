DOMAIN=$1
for SPLIT in train dev test; do \
	python -m examples.roberta.multiprocessing_bpe_encoder \
	        --encoder-json gpt2_bpe/encoder.json \
		--vocab-bpe gpt2_bpe/vocab.bpe \
		--inputs ${DOMAIN}/${SPLIT}.txt \
                --outputs ${DOMAIN}/${SPLIT}.txt.bpe \
                --keep-empty \
		--workers 60; \
done
