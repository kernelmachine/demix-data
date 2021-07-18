

# for domain in openwebtext med realnews reviews legal cs; do
#     python -m domain_loader.make_splits --domain $domain --num-workers 16 --batch-size 1024 --output-dir ../raw_data/${domain}/splits-clustered-final-eval/ --load ./preinitialized_clusters_constrained/ --train-files ../raw_data/${domain}/splits-final/train_files.txt --dev-only;
#     python -m domain_loader.make_splits --domain $domain --num-workers 16 --batch-size 1024 --output-dir ../raw_data/${domain}/splits-clustered-final-eval/ --load ./preinitialized_clusters_constrained/ --train-files ../raw_data/${domain}/splits-final/train_files.txt --dev-files ../raw_data/${domain}/splits-clustered-final-eval/dev_files.txt --test-only;
# done



for domain in 1b reddit; do
    python -m domain_loader.make_splits --domain $domain --num-workers 1 --batch-size 1 --output-dir ../raw_data/${domain}/splits-clustered-final-eval/ --load ./preinitialized_clusters_constrained/ --train-files ../raw_data/${domain}/splits-final/train_files.txt --dev-only;
    python -m domain_loader.make_splits --domain $domain --num-workers 1 --batch-size 1 --output-dir ../raw_data/${domain}/splits-clustered-final-eval/ --load ./preinitialized_clusters_constrained/ --train-files ../raw_data/${domain}/splits-final/train_files.txt --dev-files ../raw_data/${domain}/splits-clustered-final-eval/dev_files.txt --test-only;
done