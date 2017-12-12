# Artificial Painter
Mini-project for Computational Creativity course in University of Helsinki.

The main goal is to create novel art with Generative Adversarial Network. It uses a corpus of images to train the neural network to generate "human-like" art.

To run the code please clone the repository and follow the instructions in installation.txt

If you want to check that the generated images are not too similar with the original paintings you can run 
`python checkoverfit.py <folder of generated images>`.
Similar items are then written on "similar_images.csv"

Based on adversarial models cloned from https://github.com/bstriner/keras-adversarial.git
Images for training are from Web Gallery of Art https://www.wga.hu/

