import newspaper
import random
import pandas as pd
from tqdm.auto import tqdm
import argparse
from pathlib import Path
from collections import defaultdict

def parse(paper, i):
    article = paper.articles[i]
    article.download()
    try:
        article.parse()
    except Exception as e:
        print(e)
        return
    return {'text': article.text, 'title': article.title, 'paper': paper.url, 'url': article.url}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-articles-per-source', type=int, help='number of articles to fetch per source')
    parser.add_argument('--path-to-output', type=Path, help='path to output file')
    args = parser.parse_args()
    outputs = []
    papers = []
    # build papers
    df = pd.read_csv("corpus.tsv", sep='\t')
    sources = df.loc[df.fact == 'high'].source_url
    for source in tqdm(sources, desc="building sources"):
        try:
            papers.append(newspaper.build(source, language='en', memoize_articles=False))
        except:
            print(f"could not build {source}, skipping...")
            continue
    # parse downloaded articles
    errors = defaultdict(int)
    for paper in tqdm(papers, desc='parsing articles'):
        try:
            random_indexes = random.choices(range(paper.size()), k=args.num_articles_per_source)
            for i in tqdm(random_indexes):
                output = parse(paper, i)
                if output:
                    outputs.append(output)
                else:
                    errors[paper.url] += 1
        except:
            continue

    pd.DataFrame(outputs).to_json(args.path_to_output, lines=True, orient='records')
    if not errors:
        print(f"Completed. No errors!")
    else:
        print(f"Completed. Errors: {dict(errors)}")
