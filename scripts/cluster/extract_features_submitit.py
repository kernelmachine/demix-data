import argparse
import json
import logging
import os
import sys
from typing import Iterator, Iterable, TypeVar, List
from itertools import islice
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import submitit
import numpy as np
import pandas as pd
import time
from pathlib import Path
from fairseq.data.encoders.gpt2_bpe import get_encoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break

def get_json_data(input_file, predictor=None, ids_already_done=[]):
    if ids_already_done:
        set_ids_already_done = {x : 1 for x in ids_already_done}
    else:
        set_ids_already_done = {}
    if input_file == "-":
        for line in sys.stdin:
            if not line.isspace():
                if predictor:
                    res = predictor.load_line(line)
                    if not set_ids_already_done.get(res['id']):
                        yield res
                else:
                    res = json.loads(line)
                    if not set_ids_already_done.get(res['id']):
                        yield res
    else:
        with open(input_file, "r") as file_input:
            for line in tqdm(file_input):
                if not line.isspace():
                    if predictor:
                        res = predictor.load_line(line)
                        if not set_ids_already_done.get(res['id']):
                            yield res
                    else:
                        res = json.loads(line)
                        if not set_ids_already_done.get(res['id']):
                            yield res

def predict_json(predictor, batch_data):
    if len(batch_data) == 1:
        results = [predictor.predict_json(batch_data[0])]
    else:
        results = predictor.predict_batch_json(batch_data)
    for output in results:
        yield output

def extract(input_df, output_file, model_path, batch_size, submitit=False):
    if submitit:
        job_env = submitit.JobEnvironment()
    if  model_path == 'tfidf':
        model = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', min_df=3)), 
                          ('svd', TruncatedSVD(n_components=50))])
    else:

        model = AutoModel.from_pretrained(model_path)
        if submitit:
            model = model.cuda(job_env.global_rank)
        else:
            model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.pad_token = tokenizer.eos_token
    bpe = get_encoder("/private/home/suching/raw_data/demix_scale/data-bin/encoder.json", "/private/home/suching/raw_data/demix_scale/data-bin/vocab.bpe")

    input_df['text'] = input_df.text.apply(bpe.decode)

    
    df_length = input_df.shape[0]
    if os.path.exists(output_file):
        ids_already_done, vectors_already_done = torch.load(output_file)
    else:
        ids_already_done = torch.IntTensor()
        vectors_already_done = torch.FloatTensor()
    df_iterator = lazy_groups_of(input_df.to_dict(orient='records'), batch_size)
    ids = []
    if  model_path == 'tfidf':
        all_lines = []
    for batch_json in tqdm(df_iterator,
                            total=df_length // batch_size):
        lines = [x['text'] for x in batch_json if x['text'] and x['text'] != '\n']
        if  model_path == 'tfidf':
            all_lines.extend(lines)
            
            # vectors_ = TruncatedSVD().transform(vectors_)
            # vectors.append(torch.FloatTensor(vectors_))
        else:
            input_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
            last_non_masked_idx = torch.sum(input_ids['attention_mask'], dim=1) - 1
            # last_non_masked_idx = last_non_masked_idx.view(-1, 1).repeat(1, 768).unsqueeze(1).cuda()
            input_ids = input_ids.to(model.device)

            with torch.no_grad():
                vectors_ = model(**input_ids)
                #vs = []
                # for ix, idx in enumerate(last_non_masked_idx):
                    # vs.append(out[0][ix, :idx, :])
                vectors.append((vectors_[0].sum(axis=1) / input_ids['attention_mask'].sum(axis=-1).unsqueeze(-1)))
    #            for ix, idx in enumerate(last_non_masked_idx):
    #                vs.append(torch.mean(vectors_[0][ix, :idx, :], 0).unsqueeze(0)) # Models outputs are now tuples
                #vectors_ = out[0].gather(1, last_non_masked_idx.to(out[0].device)).squeeze(1) # Models outputs are now tuples
                #vectors_ = torch.cat(vs, 0)
            #vectors.append(torch.mean(vectors_[0], 1).cpu())
        indices = torch.IntTensor([x['id'] for x in batch_json]).unsqueeze(-1)
        ids.append(indices)
    if  model_path == 'tfidf':
        vectors_ = model.fit_transform(tqdm(all_lines))
    else:
        vectors_ = torch.cat(vectors, 0).cpu().numpy()

    ids_ = torch.cat(ids, 0).cpu().numpy()
    # 

    # x = np.memmap(output_file, mode='w+', dtype=np.float32, shape=(3,)


    # # torch.save((torch.cat([ids_already_done, ids_], 0),
    #             torch.cat([vectors_already_done, vectors_], 0)), output_file)
    # torch.cat(vectors, 0).cpu().numpy()
    vecs = np.save(str(output_file) + '.vecs.npy', vectors_)
    ids = np.save(str(output_file) + '.ids.npy', ids_)
    # np.save((ids.shape[0], torch.cat(ids, 0).cpu().numpy(), ), output_file)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="path to huggingface embedding model name (e.g. roberta-base) or tfidf for tfidf vectorization")
    parser.add_argument('--seq-len', type=int, required=True, help="sequence length of embedding model")
    parser.add_argument("--output_dir", type=Path, required=True, help='path to output')
    parser.add_argument("--input_file", type=str, required=True, help='path to output')
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    num_shards = parser.add_argument('--num_shards', type=int, required=False, default=10)
    parser.add_argument('--log_dir', type=str, required=False, default="log_test")
    parser.add_argument("--submitit", action='store_true')
    
    args = parser.parse_args()

    tfidf = args.model == "tfidf"
    if args.num_shards > 1 and tfidf:
        raise argparse.ArgumentError(num_shards, "Cannot perform tfidf on more than one shard")

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    vectors = []
    ids = []
    
    print(f'reading data from {args.input_file}...')

    with open(args.input_file, 'r') as f:
        z = f.read()

    text = [int(x.strip()) for x in z.split()]
    text = list(chunks(text, args.seq_len))

    df = pd.DataFrame({"text": text, "id": range(len(text))})
    #df = pd.read_json(args.input_file, lines=True)
    
    
    if args.submitit:
        df_chunks = np.array_split(df, args.num_shards)
        output_files = [args.output_dir  / f"{ix}.npy" for ix in range(len(df_chunks))]    

        log_folder = f"{args.log_dir}/%j"
    
        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(slurm_array_parallelism=args.num_shards,
                               timeout_min=60,
                               slurm_partition="learnlab,devlab",
                               cpus_per_task=10,
                               tasks_per_node=1,
                               gpus_per_node=1)
        jobs = executor.map_array(extract,
                              df_chunks,
                              output_files,
                              [args.model] * len(df_chunks),
                              [args.batch_size] * len(df_chunks))
        print('submitted job!')
        num_finished = sum(job.done() for job in jobs)
        pbar = tqdm(total=len(jobs))
        pbar.set_description('job completion status')

        while num_finished < len(jobs):
            # wait and check how many have finished
            time.sleep(5)   
            curr_finished = sum(job.done() for job in jobs)
            if curr_finished != num_finished:
                num_finished = curr_finished
                pbar.update(1)
        pbar.close()
    else:
        extract(df, args.output_dir / f"{0}.pt", args.model, args.batch_size)
