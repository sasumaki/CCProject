import csv
import os
from multiprocessing.pool import Pool
from typing import List

import requests
from collections import defaultdict


def load_catalog() -> List[List[str]]:
    with open('catalog.csv') as f:
        reader = csv.reader(f, delimiter=';')
        return list(reader)[1:]


def art_paths(catalog: List[List[str]]) -> List[str]:
    return [row[6] for row in catalog]


def download_image(art_path: str):
    converted_path = art_path.replace("/html/", "/detail/").replace(".html", ".jpg")
    return requests.get(converted_path).content


def download(art_paths_with_idx):
    for idx, image in art_paths_with_idx:
        image_file = download_image(image)
        with open(os.path.join('images', str(idx) + '.jpg'), 'wb') as f:
            try:
                f.write(image_file)
            except TypeError as e:
                pass


if __name__ == '__main__':
    catalog = load_catalog()
    split_paths = defaultdict(list)
    for idx, path in enumerate(art_paths(catalog)):
        split_paths[idx % 100].append((idx, path))

    p = Pool(100)
    p.map(download, split_paths.values())

