import glob
import os
import pickle

import numpy as np
from matplotlib.pyplot import imread

from fetchImages import catalog
from fetchImages.catalog import load_catalog
from fetchImages.image_preprocess import sort_by_image_index


def load_training_data():
    """
    Loads 64x64 images as NumPy arrays from the images64x64.
    :return:
    """
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
    """
    Returns indexes of all images that are paintings.
    :return: Indexes of all images that are paintings as a list.
    """
    all_forms = catalog.forms(load_catalog())
    paintings = [idx for idx, form in enumerate(all_forms) if form == 'painting']
    return paintings


def filter_paintings(images):
    return images[painting_filter()]


def preprocess(images):
    """
    Does nothing at the moment.
    """
    return images

if __name__ == '__main__':
    load_training_data()
