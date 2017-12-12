# Artificial Painter
Mini-project for Computational Creativity course in University of Helsinki.

The main goal is to create novel art with Generative Adversarial Network. It uses a corpus of images to train the neural network to generate "human-like" art.

To run the code please clone the repository and follow the instructions in installation.txt

If you want to check that the generated images are not too similar with the original paintings you can run 
`python checkoverfit.py <folder of generated images>`.
Similar items are then written on "similar_images.csv"

Based on adversarial models cloned from https://github.com/bstriner/keras-adversarial.git
Images for training are from Web Gallery of Art https://www.wga.hu/

## Installation
TO RUN THIS CODE YOU NEED 
PYTHON 3.6 (64-bit) and TensorFlow with CUDA® Toolkit 8.0 and cuDNN v6.1 properly installed. Also a GPU card with CUDA Compute Capability 3.0 or higher.
https://www.tensorflow.org/install
Please note that everything was done in Windows 10 operating system and is NOT tested on any other operating system.

1. run "pip install -r requirements.txt --upgrade"
2. Install adversarial_keras by navigating to the "adversarial" folder and running "python setup.py install".
3. If you already have images in your "fetchImages/images64x64" folder you can skip steps 4. and 5.
4. Download the images from The Web Gallery of Art by running "python fetch_images.py" in the "fetchImages" folder.
5. Preprocess the images to 64x64 size by "running image_preprocess.py" in the "fetchImages" folder.
6. run "python nnetwork.py"
