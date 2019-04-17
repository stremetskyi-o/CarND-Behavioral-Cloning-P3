import numpy as np
from keras import Sequential, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Cropping2D, Lambda, Conv2D, Flatten
from math import ceil

from dataset import Dataset


def resize(img):
    dsize = (66, 200)
    import tensorflow as tf
    return tf.image.resize_images(img, dsize)


if __name__ == '__main__':
    # Load and prepare dataset
    ds_shape = (160, 320, 3)
    dataset = Dataset('data')
    dataset.load()
    print('Dataset: size =', len(dataset.imgs), 'shape =', ds_shape)
    labels = np.array(dataset.labels)
    straight_label_count = np.count_nonzero(np.abs(labels) <= 0.03)
    print('Labels: mean =', np.mean(labels), 'std =', np.std(labels), 'straight ratio =',
          straight_label_count / labels.size)
    dataset.augment()
    print('Dataset size after augmentation =', len(dataset.imgs))
    dataset.validation_split(0.2)
    print('Train size=', len(dataset.x_train))
    print('Validation size=', len(dataset.x_valid))

    # Define the model
    model = Sequential()
    model.add(Cropping2D(cropping=((65, 24), (0, 0)), input_shape=ds_shape))
    model.add(Lambda(resize))
    model.add(Lambda(lambda x: x / 128 - 1))
    model.add(Conv2D(24, 5, strides=2, activation='relu'))
    model.add(Conv2D(36, 5, strides=2, activation='relu'))
    model.add(Conv2D(48, 5, strides=2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    # Load the model
    # model = models.load_model('model-bkp.h5')
    # for layer in model.layers[:9]:
    #     layer.trainable = False
    # model.summary()

    # Train the model
    batch_size = 32
    epochs = 15  # 5 for retraining
    checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(min_delta=0.0001, patience=2, verbose=1)

    model.compile('adam', 'mse')
    model.fit_generator(dataset.train_set(batch_size),
                        steps_per_epoch=ceil(len(dataset.x_train) / batch_size),
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping],
                        validation_data=dataset.valid_set(batch_size),
                        validation_steps=ceil(len(dataset.x_valid) / batch_size))
