import os
import argparse
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Input, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

from helper.preprocessing_2d import ImageDataGenerator


def get_data_files(root):
    for item in os.scandir(root):
        if item.is_file() and item.path.endswith('.npy'):
            yield item.path
        elif item.is_dir():
            yield from get_data_files(item.path)


def create_data_generators(train_dir, val_dir, image_shape, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir, image_shape=image_shape, batch_size=batch_size, class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, image_shape=image_shape, batch_size=batch_size, class_mode='binary'
    )

    return train_generator, val_generator


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


def define_model(image_shape):
    img_input = Input(shape=image_shape)

    x = Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='same')(img_input)

    x = res_block(x, nb_filters=128, block=0, subsample_factor=1)
    x = res_block(x, nb_filters=128, block=0, subsample_factor=1)
    x = res_block(x, nb_filters=128, block=0, subsample_factor=1)

    x = res_block(x, nb_filters=256, block=1, subsample_factor=2)
    x = res_block(x, nb_filters=256, block=1, subsample_factor=1)
    x = res_block(x, nb_filters=256, block=1, subsample_factor=1)

    x = res_block(x, nb_filters=512, block=2, subsample_factor=2)
    x = res_block(x, nb_filters=512, block=2, subsample_factor=1)
    x = res_block(x, nb_filters=512, block=2, subsample_factor=1)

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(img_input, x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
    model.summary()
    return model


def train(model, train_generator, val_generator, weights_filepath, config):
    (batch_size, nb_epoch, samples_per_epoch, nb_val_samples) = config

    checkpointer = ModelCheckpoint(filepath=weights_filepath, verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    model.fit_generator(train_generator, samples_per_epoch, nb_epoch,
                        verbose=2, callbacks=[checkpointer, early_stopping],
                        validation_data=val_generator, nb_val_samples=nb_val_samples,
                        class_weight=None, max_q_size=20, nb_worker=4,
                        pickle_safe=False, initial_epoch=0)


if __name__ == '__main__':
    SETTINGS_FILE_PATH = '../../SETTINGS.json'
    BASEPATH = os.path.dirname(os.path.abspath(SETTINGS_FILE_PATH))

    with open(SETTINGS_FILE_PATH, 'r') as f:
        SETTINGS = json.load(f)

    # preprocessed data config
    spacing = '1'
    dimensions = '2d'
    patchsize = 32
    scaling = 'stretch'
    multiple = 'largest'
    padding = 0
    offcenter = 25
    rotation = 0
    view = 'axial'

    image_shape = (patchsize, patchsize, 1)

    preprocessed_data_config_str = '{}mm_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        spacing, dimensions, patchsize, scaling, multiple, padding, offcenter, rotation, view
    )

    train_dir = os.path.join(BASEPATH, 'data_train/stage1/cancer_pred_anno/patches_{}/train'.format(preprocessed_data_config_str))
    val_dir = os.path.join(BASEPATH, 'data_train/stage1/cancer_pred_anno/patches_{}/val'.format(preprocessed_data_config_str))
    weights_filepath = os.path.join(BASEPATH, 'weights/stage1/cancer_pred_anno/{}'.format(
        os.path.basename(__file__).replace('.py', '.hdf5')
    ))
    os.makedirs(os.path.join(BASEPATH, 'weights/stage1/cancer_pred_anno'), exist_ok=True)

    # training config
    batch_size = 16
    nb_epoch = 1000
    samples_per_epoch = batch_size * (len(list(get_data_files(train_dir))) // batch_size)
    nb_val_samples = batch_size * (len(list(get_data_files(val_dir))) // batch_size)
    training_config = (
        batch_size, nb_epoch, samples_per_epoch, nb_val_samples
    )

    print('preprocessed data config\n-------------------')
    print('spacing:', spacing)
    print('dimensions:', dimensions)
    print('patchsize:', patchsize)
    print('scaling:', scaling)
    print('multiple:', multiple)
    print('padding:', padding)
    print('offcenter:', offcenter)
    print('rotation:', rotation)
    print('view:', view)
    print('training config\n-------------------')
    print('batch_size:', batch_size)
    print('samples_per_epoch:', samples_per_epoch)
    print('nb_epoch:', nb_epoch)
    print('nb_val_samples:', nb_val_samples)
    print('paths\n-------------------')
    print('train dir:', train_dir)
    print('val dir:', val_dir)
    print('weights filepath:', weights_filepath)
    print('# train(-):', len(list(get_data_files(train_dir + '/0'))))
    print('# train(+):', len(list(get_data_files(train_dir + '/1'))))
    print('# val(-):', len(list(get_data_files(val_dir + '/0'))))
    print('# val(+):', len(list(get_data_files(val_dir + '/1'))))
    print('\n')

    train_generator, val_generator = create_data_generators(train_dir, val_dir, image_shape, batch_size)
    model = define_model(image_shape)
    train(model, train_generator, val_generator, weights_filepath, training_config)
