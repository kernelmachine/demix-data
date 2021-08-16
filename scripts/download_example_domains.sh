for domain in ag amazon imdb chemprot rct-20k hyperpartisan_news; do
    if [ ! -d "example_domains/$domain/" ]; then
        echo "Processing $domain"
        mkdir example_domains/$domain/
        curl -Lo example_domains/$domain/train.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/train.jsonl;
        curl -Lo example_domains/$domain/dev.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/dev.jsonl;
        curl -Lo example_domains/$domain/test.jsonl https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/$domain/test.jsonl;
        cat example_domains/$domain/train.jsonl example_domains/$domain/dev.jsonl example_domains/$domain/test.jsonl > example_domains/$domain/$domain.jsonl;
    else echo "$domain already exists."
    fi;
done;
