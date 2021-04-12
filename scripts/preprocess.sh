DOMAIN=$1
fairseq-preprocess \
	    --only-source \
	           --srcdict gpt2_bpe/dict.txt \
		    --trainpref ${DOMAIN}/splits-big/train.txt.bpe \
		        --validpref ${DOMAIN}/splits-big/dev.txt.bpe\
			    --testpref ${DOMAIN}/splits-big/test.txt.bpe \
			        --destdir ${DOMAIN}/data-bin-big/ \
				    --workers 60
