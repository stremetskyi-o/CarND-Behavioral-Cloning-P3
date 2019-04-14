from keras import Input, Model
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Cropping2D
from math import ceil

from dataset import Dataset

if __name__ == '__main__':
    # Load and prepare dataset
    ds_shape = (160, 320, 3)
    dataset = Dataset('data')
    dataset.load()
    print('Dataset size =', len(dataset.imgs), 'shape =', ds_shape)
    dataset.augment()
    print('Dataset size after augmentation =', len(dataset.imgs))
    dataset.validation_split(0.2)
    print('Train size=', len(dataset.x_train))
    print('Validation size=', len(dataset.x_valid))

    # Define the model
    model_input = Input(shape=ds_shape)
    model_input_crop = Cropping2D(cropping=((65, 24), (0, 0)))(model_input)

    xception = Xception(include_top=False, weights=None, pooling='avg')

    model_output = xception(model_input_crop)
    model_output = Dense(1)(model_output)

    model = Model(model_input, model_output)
    model.summary()

    # Train the model
    batch_size = 64
    checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

    model.compile('adam', 'mse')
    model.fit_generator(dataset.train_set(batch_size),
                        steps_per_epoch=ceil(len(dataset.x_train) / batch_size),
                        epochs=3,
                        callbacks=[checkpoint],
                        validation_data=dataset.valid_set(batch_size),
                        validation_steps=ceil(len(dataset.x_valid) / batch_size))
