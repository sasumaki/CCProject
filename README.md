# Artificial Painter
Mini-project for Computational Creativity course in University of Helsinki.

The main goal is to create novel art with Generative Adversarial Network. It uses a corpus of images to train the neural network to generate "human-like" art.

To run the code please clone the repository and follow the instructions in installation.txt

If you want to check that the generated images are not too similar with the original paintings you can run 
`python checkoverfit.py <folder of generated images>`.
Similar items are then written on "similar_images.csv"

Based on adversarial models cloned from https://github.com/bstriner/keras-adversarial.git
Images used for training are from the [Web Gallery of Art](https://www.wga.hu/)

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

## Project description
[How to Build a CC System
Paper type: System Description Paper (2017) by Dan Ventura](http://computationalcreativity.net/iccc2017/ICCC_17_accepted_submissions/ICCC-17_paper_20.pdf) proposes a way of modeling computationally creative systems. Our system has many similarities to the proposed model, but it is different in some matters. Our domain is images, more specifically artistic paintings. 

Regular 64x64 pixel digital images are the phenotypic representation of our domain. Internally, we use three 64x64 NumPy arrays to represent a single image. Each one of the three arrays correspond to one of the RGB channels of the image. Moreover, the RGB values are divided by 255 since neural networks work better with such representations.

Our knowledge base consists of over 30,000 resized and cropped labeled paintings originally from the [Web Gallery of Art](https://www.wga.hu/). The resizing was done for the sake of training speed since [generative adversarial networks (GANs)](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) have a reputation of being slow to train.

The GAN consists of a generator and a discriminator. The generator is responsible for producing human-like paintings and the discriminator tries to differentiate between human- and machine-made paintings. In GANs the generator is the conceptualization and the discriminator is a genotypic evaluator. However, the generator never has a direct access to the knowledge base. Instead, the knowledge comes indirectly through the evaluator (discriminator) that has learned some desireable characteristics (aesthetics) of paintings from the knowledge base. This pattern is not directly present in Ventura's model. What is also missing is that the evaluator learns continuously from the newly generated genotypes and starts to eventually "dislike" too similar paintings. This encourages the generator to create novel genotypes.

Instead of thinking that just the end product of training is creative, we think that it is actually the training process that is the most creative part of this system as the discriminator remembers previous generated artifacts. This feature makes the training process metacreative.
