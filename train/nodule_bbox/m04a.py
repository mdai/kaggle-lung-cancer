import os
import argparse
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input, merge, Lambda
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K


def res_block(input_tensor, nb_filters=16, block=0, subsample_factor=1):
    subsample = (subsample_factor, subsample_factor)

    x = BatchNormalization(axis=3)(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters, 3, 3, subsample=subsample, border_mode='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters, 3, 3, subsample=(1, 1), border_mode='same')(x)

    if subsample_factor > 1:
        shortcut = Convolution2D(nb_filters, 1, 1, subsample=subsample, border_mode='same')(input_tensor)
    else:
        shortcut = input_tensor

    x = merge([x, shortcut], mode='sum')
    return x


def define_model(image_shape, transfer_weights_filepath):
    img_input = Input(shape=image_shape)

    x = Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same')(img_input)

    x = res_block(x, nb_filters=32, block=0, subsample_factor=1)
    x = res_block(x, nb_filters=32, block=0, subsample_factor=1)
    x = res_block(x, nb_filters=32, block=0, subsample_factor=1)

    x = res_block(x, nb_filters=64, block=1, subsample_factor=2)
    x = res_block(x, nb_filters=64, block=1, subsample_factor=1)
    x = res_block(x, nb_filters=64, block=1, subsample_factor=1)

    x = res_block(x, nb_filters=128, block=2, subsample_factor=2)
    x = res_block(x, nb_filters=128, block=2, subsample_factor=1)
    x = res_block(x, nb_filters=128, block=2, subsample_factor=1)
    x = res_block(x, nb_filters=128, block=2, subsample_factor=1)

    x = res_block(x, nb_filters=256, block=3, subsample_factor=2)
    x = res_block(x, nb_filters=256, block=3, subsample_factor=1)
    x = res_block(x, nb_filters=256, block=3, subsample_factor=1)
    x = res_block(x, nb_filters=256, block=3, subsample_factor=1)

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(x)
    x = Flatten()(x)
    x_orig = Dense(1, activation='sigmoid')(x)

    model_base = Model(img_input, x_orig)
    model_base.load_weights(transfer_weights_filepath)

    bbox = Dense(4, activation='linear', name='bbox')(x)
    model_bbox = Model(img_input, bbox)
    model_bbox.compile(optimizer='adam', loss='mae')
    model_bbox.summary()
    return model_bbox


def train(model, data_train, data_val, weights_filepath, config):
    patches_train, bboxes_train = data_train
    patches_val, bboxes_val = data_val
    batch_size, nb_epoch = config

    checkpointer = ModelCheckpoint(filepath=weights_filepath, verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    model.fit(patches_train, bboxes_train,
              validation_data=(patches_val, bboxes_val),
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=2, callbacks=[checkpointer, early_stopping], shuffle=True)


if __name__ == '__main__':
    SETTINGS_FILE_PATH = '../../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    image_shape = (32, 32, 2)

    data_dir = os.path.join(BASEPATH, 'data_train/stage1/nodule_bbox/patches_2')
    transfer_weights_filepath = os.path.join(BASEPATH, 'weights/stage1/nodule_detect/m04a.hdf5')
    weights_filepath = os.path.join(BASEPATH, 'weights/stage1/nodule_bbox/{}'.format(
        os.path.basename(__file__).replace('.py', '.hdf5')
    ))
    os.makedirs(os.path.join(BASEPATH, 'weights/stage1/nodule_bbox'), exist_ok=True)

    patches_train = np.load(os.path.join(data_dir, 'patches_train.npy'))
    bboxes_train = np.load(os.path.join(data_dir, 'bboxes_train.npy'))
    data_train = patches_train, bboxes_train

    patches_val = np.load(os.path.join(data_dir, 'patches_val.npy'))
    bboxes_val = np.load(os.path.join(data_dir, 'bboxes_val.npy'))
    data_val = patches_val, bboxes_val

    # training config
    batch_size = 256
    nb_epoch = 1000
    training_config = (batch_size, nb_epoch)

    print('batch_size:', batch_size)
    print('nb_epoch:', nb_epoch)
    print('data train:', patches_train.shape, bboxes_train.shape)
    print('data val:', patches_val.shape, bboxes_val.shape)
    print('weights filepath:', weights_filepath)

    model = define_model(image_shape, transfer_weights_filepath)
    train(model, data_train, data_val, weights_filepath, training_config)
