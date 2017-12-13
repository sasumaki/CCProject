import imagehash
from PIL import Image
import os
import glob
from operator import itemgetter
from matplotlib.pyplot import imread
import numpy as np
import re
from pathlib import Path
import pandas as pd
from itertools import *
import sys

from fetchImages.image_preprocess import sort_by_image_index


def find_similar_images(arg):
    """
    #Uses phash to find similar images in corpus and generated images
    #https://en.wikipedia.org/wiki/Perceptual_hashing
    """
    generated_images = glob.glob(arg +"/*.png")
    hashfile = Path("corpus-hashes.csv")
    if not hashfile.is_file():
        print("Hashvalues for corpus are not precomputed.")
        corpus = generate_corpus_hashes()
    corpus = pd.read_csv(hashfile)
    print(corpus)
    targets = []
    sources = []
    dissimilarities = []
    iterator = 0
    last_operation = len(corpus) * len(generated_images)
    for image in generated_images:
        most_similar = "" 
        minsim = 10000
        target = Image.open(image)
        targethash = imagehash.phash(target)
        for hashe, source in zip(corpus["hash"], corpus["image"]):                           
            dissimilarity = imagehash.hex_to_hash(hashe) - targethash
            print(str(iterator) + " / " + str(last_operation))
            iterator = iterator + 1
            if(dissimilarity < minsim):
                minsim = dissimilarity
                most_similar = source      
        sources.append(most_similar)
        targets.append(image)
        dissimilarities.append(dissimilarity)
                
    pd.DataFrame(data= {"source": sources, "target": targets, "dissimilarity": dissimilarities}, columns=["source", "target", "dissimilarity"]).to_csv("similar_images.csv")
    return()

def generate_corpus_hashes():
    """
    #Generates a csv for images in corpus. Makes future use faster.
    """
    corpus = glob.glob('fetchImages/images64x64/*.png')
    corpus.sort(key=sort_by_image_index)
       
    print("Computing hashvalues")
    hashes = []
    counter = 0
    for source_image in corpus:          
        if(counter % 100 == 0):
            print("Index of progression: " + str(counter))
        img = Image.open(source_image)
        hashvalue = imagehash.phash(img)
        img.close()
        hashes.append(str(hashvalue))
        counter = counter + 1
    pd.DataFrame(data= {"hash": hashes, "image": corpus}, columns=["hash", "image"]).to_csv("corpus-hashes.csv")
    return()

#Not in use atleast right now
def read_generated_images(arg):
    filenames = glob.glob(arg + "*.png")
    latest_epoch = "0"
    for filename in filenames:
        epoch = str(filename.split("-")[1]).split(".")[0]
        if int(epoch) > int(latest_epoch): 
            latest_epoch = epoch
    filenames = []
    filename = "output/art7/images/epoch-" + latest_epoch + ".png"
    filenames.append(filename)
    return(filenames)

if __name__ == '__main__':
   """
   For CLI usage put folder of generated images as argument
   """
    arg = sys.argv[1]
    find_similar_images(arg)
    