import os
from collections import defaultdict
from multiprocessing.pool import Pool

import requests
from typing import List, Tuple

from fetchImages.catalog import load_catalog, art_paths


def download_image(art_path: str) -> bytes:
    """
    Downloads an image given a URL to the main page of the image.
    :param art_path: A URL to the main page of the image.
    :return: A binary representation of the image.
    """
    converted_path = art_path.replace("/html/", "/detail/").replace(".html", ".jpg")
    return requests.get(converted_path).content


def download(art_paths_with_idx: List[Tuple(int, str)]):
    """
    Downloads all images and saves them to the 'images' directory.
    :param art_paths_with_idx: A list of tuples containing the image id and the path to the main page of the image.
    """
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

