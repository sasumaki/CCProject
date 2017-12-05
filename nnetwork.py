import glob
import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
import re
from keras import Input, Model
from tensorflow.python import debug as tf_debug

import imageloader
from fetchImages import catalog

mpl.use('Agg')

import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Concatenate, K
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous
from keras_adversarial.legacy import Dense, BatchNormalization, fit, l1l2, Convolution2D, AveragePooling2D
from adversarial.examples.image_utils import dim_ordering_unfix, dim_ordering_shape


def model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    h = 5

    input_z = Input(shape=(100,))
    input_label = Input(shape=(10,))

    H = Concatenate()([input_z, input_label])
    H = Dense(nch * 4 * 4, W_regularizer=reg())(H)
    H = BatchNormalization(mode=0)(H)
    H = Reshape(dim_ordering_shape((nch, 4, 4)))(H)
    H = Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg())(H)
    H = BatchNormalization(mode=0, axis=1)(H)
    H = LeakyReLU(0.2)(H)
    H = (UpSampling2D(size=(2, 2)))(H)
    H = (Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))(H)
    H = (BatchNormalization(mode=0, axis=1))(H)
    H = (LeakyReLU(0.2))(H)
    H = (UpSampling2D(size=(2, 2)))(H)
    H = (Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg()))(H)
    H = (BatchNormalization(mode=0, axis=1))(H)
    H = (LeakyReLU(0.2))(H)
    H = (UpSampling2D(size=(2, 2)))(H)
    H = (Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg()))(H)
    H = (BatchNormalization(mode=0, axis=1))(H)
    H = (LeakyReLU(0.2))(H)
    H = (UpSampling2D(size=(2, 2)))(H)
    H = (Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))(H)
    H = (Activation('sigmoid'))(H)
    return Model(inputs=[input_z, input_label], outputs=[H, input_label])


def model_discriminator():
    nch = 512
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)

    input_d = Input(shape=(64, 64, 3))
    input_aux = Input(shape=(10,))
    c1 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=(64, 64, 3))
    c2 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg())
    c3 = Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg())
    c4 = Convolution2D(int(nch), h, h, border_mode='same', W_regularizer=reg())
    c5 = Convolution2D(1, h, h, border_mode='same', W_regularizer=reg())

    H = c1(input_d)
    H = MaxPooling2D(pool_size=(2, 2))(H)
    H = LeakyReLU(0.2)(H)
    H = c2(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)
    H = LeakyReLU(0.2)(H)
    H = c3(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)
    H = LeakyReLU(0.2)(H)
    H = c4(H)
    H = AveragePooling2D(pool_size=(4, 4), border_mode='valid')(H)
    H = Flatten()(H)
    H = Concatenate()([H, input_aux])
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dense(64)(H)
    H = LeakyReLU(0.2)(H)
    H = Dense(1)(H)
    H = Activation('sigmoid')(H)
    return Model(inputs=[input_d, input_aux], outputs=H)


optimizer = AdversarialOptimizerSimultaneous()
opt_g = Adam(1e-5, decay=1e-5)
opt_d = Adam(5e-5, decay=1e-5)
nb_epoch = 500
path = os.path.join("output", "art4")
loss = 'binary_crossentropy'
latent_dim = 100

def main():
    # Uncomment this for debugging
    # sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # K.set_session(sess)

    xtrain = imageloader.preprocess(imageloader.filter_paintings(imageloader.load_training_data()))
    y = gan_targets(xtrain.shape[0])
    y[-1] -= 0.1  # 1-sided label smoothing "hack"
    z = np.random.normal(size=(xtrain.shape[0], 100))

    catalog_file = catalog.load_catalog()
    numerical_categories = catalog.transform_categorical_to_numerical(catalog.types(catalog_file))
    one_hot_length = max(numerical_categories) + 1
    targets = np.array([numerical_categories]).reshape(-1)
    one_hots = np.eye(one_hot_length)[targets]
    one_hots = one_hots[imageloader.painting_filter()]

    generators = glob.glob(path + "/**/generator.*")
    discriminators = glob.glob(path + "/**/discriminator.*")

    current_epoch = 1
    if generators:
        latest_generator = max(generators, key=os.path.getctime)
        generator = load_model(latest_generator)
        current_epoch = max(current_epoch, 1 + int(re.search("generator\.(\d*)\.h5", latest_generator).group(1)))
    else:
        generator = model_generator()
        current_epoch = max(current_epoch, 1)

    if discriminators:
        latest_discriminator = max(discriminators, key=os.path.getctime)
        discriminator = load_model(latest_discriminator)
        current_epoch = max(current_epoch, 1 + int(re.search("discriminator\.(\d*)\.h5", latest_discriminator).group(1)))
    else:
        discriminator = model_discriminator()
        current_epoch = max(current_epoch, 1)


    generator.summary()
    discriminator.summary()
    gan = simple_gan(generator=generator,
                     discriminator=discriminator,
                     latent_sampling=None)

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    callbacks = initialize_callbacks(path, generator, discriminator, latent_dim)

    history = fit(model, x=[z, one_hots, xtrain, one_hots], y=y,
                  callbacks=callbacks, nb_epoch=nb_epoch, initial_epoch=current_epoch - 1,
                  batch_size=32)


class AdversarialModelSaver(Callback):
    def __init__(self, path, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(os.path.join(self.path, "models")):
            os.makedirs(os.path.join(self.path, "models"))
        self.generator.save(os.path.join(self.path, "models", "generator.{epoch}.h5".format(epoch=epoch)))
        self.discriminator.save(os.path.join(self.path, "models", "discriminator.{epoch}.h5".format(epoch=epoch)))


def initialize_callbacks(path, generator, discriminator, latent_dim):
    def generator_sampler():
        hots = np.eye(10)[[int(t / 10) for t in range(100)]]
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        xpred = dim_ordering_unfix(generator.predict([zsamples, hots])[0]).transpose((0, 2, 3, 1))
        return xpred.reshape((10, 10) + xpred.shape[1:])

    generator_cb = ImageGridCallback(os.path.join(path, "images", "epoch-{:03d}.png"), generator_sampler, cmap=None)
    tensor_board = TensorBoard(log_dir=os.path.join(path, 'logs'), histogram_freq=0, write_graph=True,
                               write_images=True)
    model_saver = AdversarialModelSaver(path, generator, discriminator)

    callbacks = [generator_cb, tensor_board, model_saver]
    return callbacks
if __name__ == "__main__":
    main()
