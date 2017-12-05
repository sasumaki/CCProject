import os
import numpy as np 
from scipy import misc, ndimage, cluster
import re
import pathlib
import glob
import math
import scipy
import pandas as pd





def get_color_clusters(img,idx, save_images = False, clusters = 3 ):

    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, clusters)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = ''.join(chr(int(c)) for c in peak)

    if(save_images):
        c = ar.copy()
        for i, code in enumerate(codes):
            c[scipy.r_[scipy.where(vecs==i)],:] = code
        misc.imsave("images64x64_recolored/" + str(idx) + ".png", c.reshape(*shape))

    return idx, codes/255


def read_images():
    idx = 0
    color_clusters = []
    for filename in glob.glob('images64x64/*.png'): 
        try:
            image = ndimage.imread(filename, mode="RGB")
            print(image)
        except:
            print("reading failed on " + str(filename))
            continue
        print("getting colors")
        color_clusters.append(get_color_clusters(image, idx = idx))
        idx = idx + 1
        
        

    return(color_clusters)


if __name__ == '__main__':
    print("starting")
    directory = "images64x64_recolored"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    color_clusters = read_images()
    print(color_clusters)
    df = pd.DataFrame(color_clusters)
    df.to_csv("clusters.csv")

        