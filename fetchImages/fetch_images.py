import os
from multiprocessing.pool import Pool

import os
from collections import defaultdict
from multiprocessing.pool import Pool

import requests

from fetchImages.catalog import load_catalog, art_paths


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

