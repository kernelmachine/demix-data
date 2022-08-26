#!/bin/bash

for domain in ag amazon imdb chemprot rct-20k hyperpartisan_news; do
    if [ ! -d "$DATA_DIR/$domain/" ]; then
        echo "Processing $domain"
        mkdir $DATA_DIR/$domain/
        curl -Lo $DATA_DIR/$domain/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/train.jsonl;
        curl -Lo $DATA_DIR/$domain/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/dev.jsonl;
        curl -Lo $DATA_DIR/$domain/test.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/test.jsonl;
        cat $DATA_DIR/$domain/train.jsonl $DATA_DIR/$domain/dev.jsonl $DATA_DIR/$domain/test.jsonl > $DATA_DIR/$domain/$domain.jsonl;
    else echo "$domain already exists."
    fi;
done;
