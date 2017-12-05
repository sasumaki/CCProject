import glob

import numpy as np
import pickle
import os
from matplotlib.pyplot import imread
from operator import itemgetter

from fetchImages import catalog
from fetchImages.catalog import load_catalog
import re

from fetchImages.image_preprocess import sort_by_image_index


def load_training_data():
    cache = 'cache'
    if not os.path.exists(cache):
        os.makedirs(cache)
    cached_images = os.path.join(cache, 'images.pickle')

    if os.path.exists(cached_images):
        with open(cached_images, 'rb') as f:
            return pickle.load(f)


    filenames = glob.glob('fetchImages/images64x64/*.png')
    filenames.sort(key=sort_by_image_index)

    images = [imread(filename) for filename in filenames]
    images = np.array(images)
    with open(cached_images, 'wb') as f:
        pickle.dump(images, f)
    return images


def painting_filter():
    all_forms = catalog.forms(load_catalog())
    paintings = [idx for idx, form in enumerate(all_forms) if form == 'painting']
    return paintings


def filter_paintings(images):
    return images[painting_filter()]


def preprocess(images):
    return images

if __name__ == '__main__':
    load_training_data()
