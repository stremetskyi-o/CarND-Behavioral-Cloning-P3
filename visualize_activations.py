import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from matplotlib import image as mpimg

if __name__ == '__main__':
    # Visualizing intermediate activation in Convolutional Neural Networks with Keras
    # https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md

    model = models.load_model('model.h5')
    model.summary()

    images_per_row = 2

    layer_names = [layer.name for layer in model.layers[3:5]]
    layer_outputs = [layer.output for layer in model.layers[3:5]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    fdir = 'writeup-img/'
    files = glob.glob(fdir + 'act*.jpg')

    for fname in files:
        activations = activation_model.predict(mpimg.imread(fname)[None, :, :, :])
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            n_features = layer_activation.shape[-1]  # Number of features in the feature map
            size_h = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
            size_w = layer_activation.shape[2]
            n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
            display_grid = np.zeros((size_h * n_cols, images_per_row * size_w))
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    if channel_image.std() != 0:
                        channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size_h: (col + 1) * size_h, row * size_w: (row + 1) * size_w] = channel_image
            img = cv2.resize(display_grid, (480, int(480 / display_grid.shape[1] * display_grid.shape[0])),
                             interpolation=cv2.INTER_NEAREST)
            fpath = Path(fname)
            fpath = fdir + 'out_' + fpath.stem + '_' + layer_name + fpath.suffix
            plt.imsave(fpath, img, cmap='viridis')
