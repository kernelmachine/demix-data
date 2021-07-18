from tqdm import tqdm
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        count = max([int(x.split('/')[-1]) for x in os.listdir(args.output_dir)])
    else:
        count = 0

    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.file, 'r') as f:
        z = f.read()

    texts = z.split('<|endoftext|>')[:1000000]

    

    for ix, text in tqdm(enumerate(texts), total=len(texts)):
        if ix % 512 == 0:
            count += 1
        if not os.path.isdir(f"{args.output_dir}/{count}"):
            os.mkdir(f"{args.output_dir}/{count}")
        with open(f"{args.output_dir}/{count}/{ix}.txt", 'w+') as f:
            f.write(text)
    
