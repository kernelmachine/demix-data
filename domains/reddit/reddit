for f in /checkpoint/parlai/tasks/meena_reddit/v1/*; do
	echo "processing ${f}...";
	file=$(basename $f);
	pigz -dc $f  | pv | parallel --pipe -q jq -rc '"<|endoftext|> " + .context + " " + .label' | pigz  > $file;
done
