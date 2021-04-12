pigz -dc All_Amazon_Review.json.gz | parallel --pipe -q jq -rc '"<|endoftext|>" + .reviewText' | pv | pigz > reviews.txt.gz
