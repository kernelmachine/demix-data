import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from domain_loader.domain_loader import Domain, read_domains, get_dataloader
from accelerate import Accelerator
import accelerate
from accelerate import notebook_launcher
from transformers import AutoTokenizer, AutoModel
from accelerate.utils import PrepareForLaunch
from torch.multiprocessing import start_processes
from pathlib import Path
import numpy as np
import subprocess
from fairseq.data import Dictionary, data_utils, indexed_dataset, iterators
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.logging.progress_bar import progress_bar
from fairseq.data import FairseqDataset, plasma_utils, TokenBlockDataset, ConcatDataset, SubsampleDataset
from cluster.cluster import load_model
import uuid
from sklearn.cluster import MiniBatchKMeans
from kmeans_pytorch import KMeans

DATA_DIR = "/private/home/suching/raw_data/"


        
class DataFrameDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.texts = df['text'].tolist()
        self.ids  = df['id'].tolist()
    
    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]

    def __len__(self):
        return len(self.texts)


class MMDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_ids, path_to_features, dataset_len):
        self.id_path = path_to_ids
        self.feature_path = path_to_features
        self.dataset_len = dataset_len
        self.feature_array = np.load(self.feature_path,
                             mmap_mode='r')
        self.id_array = np.load(self.id_path,
                             mmap_mode='r')

    def __getitem__(self, index):
        return np.array(self.id_array[index]), np.array(self.feature_array[index])

    def __len__(self):
        return self.dataset_len


def load_fairseq_binary(path_to_data, split, batch_size=512, tokens_per_sample=512):
    dictionary = Dictionary.load("/private/home/suching/raw_data/data-bin-big/dict.txt")
    dataset = data_utils.load_indexed_dataset(
                path_to_data,
                dictionary,
                dataset_impl="mmap",
            )
    dataset = maybe_shorten_dataset(
        dataset,
        split,
        "",
        "truncate",
        tokens_per_sample,
        42,
    )

    dataset = TokenBlockDataset(
        dataset,
        dataset.sizes,
        block_size=tokens_per_sample,
        pad=dictionary.pad(),
        eos=dictionary.eos(),
        break_mode="eos",
        include_targets=False,
    )
    dataset = MonodomainDataset(dataset, dataset.sizes, dictionary, pad_to_bsz=batch_size)
    return dataset


def extract_features_(model,
                        batch_size,
                        output_path,
                        domains=None,
                        path_to_file=None,
                        files=None,
                        sample=None,
                        path_to_data_bins=[],
                        train_clusters=False,
                        num_clusters=8,
                        save_clusters="model.pkl",
                        predict_clusters=False,
                        path_to_clusters=None,
                        path_to_output_dir=None):
    accelerator = Accelerator()
    
    if train_clusters:
        if torch.distributed.get_rank() == 0:
            kmeans = KMeans(num_clusters=num_clusters, device=accelerator.device)
        accelerator.wait_for_everyone()
    elif predict_clusters:
        kmeans = KMeans(device=accelerator.device).load(Path(path_to_clusters))
    
    if path_to_data_bins:
        datasets = []
        for path in tqdm(path_to_data_bins):
            accelerator.print(f"loading binary at {path}")
            datasets.append(load_fairseq_binary(path, "test", batch_size))
        dataset = ConcatDataset(datasets)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 collate_fn=dataset.collater,
                                                 num_workers=16,
                                                 batch_size=batch_size)
    elif path_to_file:
        texts = pd.read_csv(path_to_file, sep='\t')
        dataset = DataFrameDataset(texts)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=16, batch_size=batch_size)
    elif domains:
        dataloader = get_dataloader(domains, files=files, sample=sample, num_workers=16, batch_size=batch_size)
    else:
        raise ValueError("One of domain, path_to_file, or path_to_data_bins must be supplied.")
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)
    features = []
    ids = []
    counter = 0
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        if path_to_data_bins:
            id = batch['id']
        elif path_to_file:
            id = batch[0]
            text = batch[1]
        else:
            id = batch[0]
            filenames = batch[1]
            text = batch[2]
        if path_to_data_bins:
            input_ids = {'input_ids': batch['net_input']['src_tokens']}
        else:
            input_ids = tokenizer.batch_encode_plus(list(text),
                        add_special_tokens=True,
                        truncation=True,
                        max_length=512,
                        padding='max_length',
                        return_tensors='pt').to(accelerator.device)
        with torch.no_grad():
            out = model(**input_ids)
            feats = torch.mean(out[0], 1)
            if predict_clusters:
                clusters = kmeans.predict(feats.to(accelerator.device), distance='euclidean', balanced=True)
                for cluster, fname in zip(clusters, filenames):
                    Path(Path(path_to_output_dir) / f"{cluster}").mkdir(parents=True, exist_ok=True)
                    with open(Path(path_to_output_dir) / f"{cluster}" / f"{str(torch.distributed.get_rank())}.txt", 'a+') as f:
                        f.write(fname + "\n")
            else:
                all_feats = accelerator.gather(feats).cpu()
                if path_to_file:
                    all_ids = accelerator.gather(id).cpu()
                if train_clusters:
                    if torch.distributed.get_rank() == 0:
                        kmeans.fit(all_feats, online=True, iter_limit=10, iter_k=counter, balanced=True, tqdm_flag=False)
                    accelerator.wait_for_everyone()
                features.append(all_feats)
                if path_to_file:
                    ids.append(all_ids)
        counter += 1
    if predict_clusters:
        return
    elif train_clusters:
        if torch.distributed.get_rank() == 0:
            kmeans.save(save_clusters)
        accelerator.wait_for_everyone()
    else:
        # extracting features
        features = torch.cat(features, 0)
        ids = torch.cat(ids, 0)
        assert len(dataset) == features.shape[0]
        assert len(dataset) == ids.shape[0]
        from fairseq import pdb; pdb.set_trace()
        if not Path(output_path).exists():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        np.save(Path(output_path) / "features.pt", features)
        np.save(Path(output_path) / "ids.pt", ids)

def initialize_slurm_distributed(num_gpus=8, master_addr="127.0.0.1", master_port=29500):
    rank = int(os.environ['SLURM_PROCID'])
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

    node_list = os.environ.get("SLURM_JOB_NODELIST")
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list]
    )
    init_method= "tcp://{host}:{port}".format(
        host=hostnames.split()[0].decode("utf-8"),
        port=os.environ['MASTER_PORT'],
    )
    torch.distributed.init_process_group(backend="nccl", init_method=init_method, world_size=num_gpus, rank=rank)

def extract_features(model,
                     batch_size,
                     num_gpus,
                     output_path,
                     domains=None,
                     path_to_file=None,
                     path_to_data_bins=[],
                     master_addr="127.0.0.1",
                     master_port=29500,
                     train_clusters=False,
                     num_clusters=8,
                     predict_clusters=False,
                     path_to_clusters=None,
                     sample=None,
                     path_to_output_dir=None):
    initialize_slurm_distributed(num_gpus=num_gpus, master_addr=master_addr, master_port=master_port)
    extract_features_(model,
                    batch_size,
                    output_path,
                    domains=domains,
                    path_to_file=path_to_file,
                    path_to_data_bins=path_to_data_bins,
                    sample=sample,
                    train_clusters=train_clusters,
                    num_clusters=num_clusters,
                    predict_clusters=predict_clusters,
                    path_to_clusters=path_to_clusters,
                    path_to_output_dir=path_to_output_dir)
