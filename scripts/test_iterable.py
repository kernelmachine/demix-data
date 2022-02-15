from pathlib import Path
from domain_loader.domain_loader import IterableDomain
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# dir_ = Path('/private/home/suching/raw_data/c4/en.noblocklist')

dir_ = Path('/private/home/suching/raw_data/reddit/reddit')


domain = IterableDomain(dir_, add_bos_token=True, anonymize=False)

dataloader = DataLoader(domain, batch_size = 1024, num_workers = 2)

num_tokens = 0
curr_file = ""
pbar = tqdm(dataloader)
text_1 = []
for idx, file, text, nt, url in pbar:
    num_tokens += nt.sum()
    if file != curr_file:
        curr_file = file[0]
    if num_tokens > 1000000:
        break
    pbar.set_description(f"{curr_file}, {num_tokens}")

domain = IterableDomain(dir_, add_bos_token=True, anonymize=False)