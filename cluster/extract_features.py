import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from domain_loader.domain_loader import Domain
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
from fairseq.data import FairseqDataset, plasma_utils, TokenBlockDataset, ConcatDataset


DATA_DIR = "/private/home/suching/raw_data/"



def filter_indices_by_size(
        indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large
        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

def get_batch_iterator(
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
    
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        return epoch_iter



def collate(samples, pad_idx, eos_idx, src_domain_idx, fixed_pad_length=None, pad_to_bsz=None):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens

    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
            "src_domain_idx": src_domain_idx
        },
        "target": target
    }


class MonodomainDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monodomain data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=None,
        add_domain_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_domain_idx=None,
        tgt_domain_idx=None,
        src_domain_token=None,
        tgt_domain_token=None,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_domain_token = add_domain_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_domain_idx = src_domain_idx
        self.tgt_domain_idx = tgt_domain_idx
        self.src_domain_token = src_domain_token
        self.tgt_domain_token = tgt_domain_token        
        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets

    def __getitem__(self, index):
        if self.targets is not None:
            # *future_target* is the original sentence
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            #
            # Left-to-right language models should condition on *source* and
            # predict *future_target*.
            # Right-to-left language models should condition on *source* and
            # predict *past_target*.
            source, future_target, past_target = self.dataset[index]
            source, target = self._make_source_target(
                source, future_target, past_target
            )
        else:
            source = self.dataset[index]
            target = []
        source, target = self._maybe_add_bos(source, target)
        res = {"id": index, "source": source, "target": target}
        return res

    def __len__(self):
        return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if (
                self.add_eos_for_other_targets
                and (("self" in self.targets) or ("past" in self.targets))
                and source[-1] != self.vocab.eos()
            ):
                # append eos at the end of source
                source = torch.cat([source, source.new([self.vocab.eos()])])

                if "future" in self.targets:
                    future_target = torch.cat(
                        [future_target, future_target.new([self.vocab.pad()])]
                    )
                if "past" in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.cat(
                        [
                            past_target.new([self.vocab.pad()]),
                            past_target[1:],
                            source[-2, None],
                        ]
                    )

            for t in self.targets:
                if t == "self":
                    target.append(source)
                elif t == "future":
                    target.append(future_target)
                elif t == "past":
                    target.append(past_target)
                else:
                    raise Exception("invalid target " + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, self._filter_vocab(target)

    def _maybe_add_bos(self, source, target):
        if self.add_domain_token:
            # src_lang_idx and tgt_lang_idx are passed in for multilingual LM, with the
            # first token being an lang_id token.
            bos = self.src_domain_token or self.vocab.bos()
            source = torch.cat([source.new([bos]), source])
            if target is not None:
                tgt_bos = self.tgt_domain_token or self.tgt_vocab.bos()
                target = torch.cat([target.new([tgt_bos]), target])
        return source, target

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    
    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(
            samples, 
            self.vocab.pad(), 
            self.vocab.eos(), 
            [self.src_domain_idx] * len(samples),
            self.fixed_pad_length,
            self.pad_to_bsz
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.RandomState(seed=torch.distributed.get_rank()).permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

        
class DataFrameDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.texts = df.text.tolist()
        self.id  = df['id'].tolist()
    
    def __getitem__(self, idx):
        return self.id[idx], self.texts[idx]

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
        break_mode="none",
        include_targets=False,
    )
    dataset = MonodomainDataset(dataset, dataset.sizes, dictionary, pad_to_bsz=batch_size, fixed_pad_length=tokens_per_sample)
    return dataset


def extract_features_( model, batch_size, fname, domain=None, path_to_file=None, files=None, sample=None, path_to_data_bins=[]):
    accelerator = Accelerator()

    if path_to_data_bins:
        datasets = []
        for path in path_to_data_bins:
            accelerator.print(f"loading binary at {path}")
            datasets.append(load_fairseq_binary(path, "test", batch_size))
        dataset = ConcatDataset(datasets)
    elif path_to_file:
        texts = pd.read_csv(path_to_data)
        dataset = DataFrameDataset(texts)
    elif domain:
        resolved_path = Path(DATA_DIR) / domain / domain
        if files:
            domain_files = files
        else:
            with open(Path(DATA_DIR) / domain  / "metadata" / "filenames.txt", 'r') as f:
                domain_files = []
                for x in tqdm(f.readlines()):
                    fp = x.strip()
                    domain_files.append(fp)
            if domain == 'reddit':
                batch_size = 1
                add_bos_token=False
            else:
                if domain != '1b':
                    add_bos_token=True
                else:
                    add_bos_token=False
                sample = sample
            if sample:
                domain_files = np.random.choice(domain_files, sample)
        dataset = Domain(resolved_path,
                        filenames=list(domain_files),
                        add_bos_token=True,
                        silent=True)
    else:
        raise ValueError("One of domain, path_to_file, or path_to_data_bins must be supplied.")

    

    if path_to_data_bins:
        data = torch.utils.data.DataLoader(dataset, 
            collate_fn=dataset.collater,
            batch_size=batch_size)
    else:
        data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModel.from_pretrained(model)

    model, tokenizer, data = accelerator.prepare(model, tokenizer, data)
    accelerator.print(not accelerator.is_local_main_process)
    features = []
    ids = []

    if path_to_data_bins:
        for batch in tqdm(data, disable=not accelerator.is_local_main_process):
            ix = batch['id']
            input_ids = {'input_ids': batch['net_input']['src_tokens']}
            with torch.no_grad():
                out = model(**input_ids)
                feats = torch.mean(out[0], 1)
                all_feats = accelerator.gather(feats).cpu()
                all_ids = accelerator.gather(ix).cpu()
                features.append(all_feats)
                ids.append(all_ids)
    elif path_to_file is not None:
        for ix, text in tqdm(data, disable=not accelerator.is_local_main_process):
            input_ids = tokenizer.batch_encode_plus(list(text),
                            add_special_tokens=True,
                        truncation=True,
                        max_length=512,
                        padding='max_length',
                        return_tensors='pt').to(accelerator.device)
            with torch.no_grad():
                out = model(**input_ids)
                feats = torch.mean(out[0], 1)
                all_feats = accelerator.gather(feats).cpu()
                all_ids = accelerator.gather(ix).cpu()
                features.append(all_feats)
                ids.append(all_ids)
    else:
        for ix, _, text, _, _ in tqdm(data, disable=not accelerator.is_local_main_process):
            input_ids = tokenizer.batch_encode_plus(list(text),
                            add_special_tokens=True,
                        truncation=True,
                        max_length=512,
                        padding='max_length',
                        return_tensors='pt').to(accelerator.device)
            with torch.no_grad():
                out = model(**input_ids)
                feats = torch.mean(out[0], 1)
                all_feats = accelerator.gather(feats).cpu()
                all_ids = accelerator.gather(ix).cpu()
                features.append(all_feats)
                ids.append(all_ids)

    features = torch.cat(features, 0)
    ids = torch.cat(ids, 0)
    np.save(fname + "_features", features)
    np.save(fname + "_ids", ids)

def extract_features(model, batch_size, num_gpus, fname, domain=None, path_to_file=None, path_to_data_bins=[], master_addr="127.0.0.1", master_port=29500, notebook=False, sample=None):
    if notebook:
        notebook_launcher(extract_features_,
                                    args=(domain,
                                        model,
                                        batch_size,
                                        fname),
                                num_processes=num_gpus)
    else:
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
        extract_features_(model, batch_size, fname, domain=domain, path_to_file=path_to_file, path_to_data_bins=path_to_data_bins, sample=sample)
