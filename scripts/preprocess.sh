DOMAIN=$1
OUTPUT_DIR=$2
fairseq-preprocess \
	    --only-source \
	           --srcdict gpt2_bpe/dict.txt \
		    --trainpref ${DOMAIN}/train.txt.bpe \
		        --validpref ${DOMAIN}/dev.txt.bpe\
			    --testpref ${DOMAIN}/test.txt.bpe \
			        --destdir ${OUTPUT_DIR}/data-bin/${DOMAIN} \
				    --workers 60
