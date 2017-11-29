
import os
import numpy as np 
from scipy import misc, ndimage
import re
import pathlib
import glob
from PIL import Image
import math



#Crops images from the center
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    
 
    return img[starty:starty+cropy,startx:startx+cropx]


#Resizes images maintaining aspect ratio (as accurate as possible)
def resize(img):
    y,x, c = img.shape
    if y < x:
        aspect = (x/y)
    else:
        aspect = (y/x)
    print(y,x,aspect)
    if y < x:
        img = misc.imresize(img, (64, math.ceil(64*aspect)))
    else:
        img = misc.imresize(img, (math.ceil(64*aspect),64))

    return img

# Calls methods to preprocess images to 64x64 size
def preprocess(images):
    
    for filename in glob.glob('images/*.jpg'): 
        try:
            image = ndimage.imread(filename, mode="RGB")
        except:
            continue
        image = resize(image)
        print(image.shape)
        image_cropped = crop_center(image,64,64)
        images.append(image_cropped)
    return images

# Preprocesses images to 64x64 size and saves in folder.
if __name__ == '__main__':
    images = []
    images = preprocess(images)
    directory = "images64x64"
    if not os.path.exists(directory):
        os.makedirs(directory)

    idx = 0
    
    for image in images:
        print("saving img " + str(idx) + "...")
        misc.imsave("images64x64/" + str(idx) + ".jpg", image)
        idx += 1
        
