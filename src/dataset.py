import csv
import os
from os import path

import numpy as np
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IMG_DIR = 'IMG'
DESC_FILE = 'driving_log.csv'


class ImageDefinition:

    def __init__(self, fname, transformation=None):
        self.fname = fname
        self.transformation = transformation

    def with_transformation(self, transformation):
        return ImageDefinition(self.fname, transformation)

    def load_image(self):
        img = mpimg.imread(self.fname)
        if self.transformation:
            img = self.transformation(img)
        return img


class Dataset:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.datasets = [path.join(base_dir, ds_dir) for ds_dir in next(os.walk(base_dir))[1]]
        self.imgs, self.labels = None, None
        self.x_train, self.y_train, self.x_valid, self.y_valid = None, None, None, None

    def load(self):
        imgs = []
        labels = []
        for dataset in self.datasets:
            desc = path.join(dataset, DESC_FILE)
            if path.exists(desc):
                with open(desc, 'r') as desc_file:
                    for row in csv.reader(desc_file):
                        imgs.append(ImageDefinition(path.join(dataset, IMG_DIR, path.basename(row[0]))))
                        labels.append(float(row[3]))
        self.imgs = imgs
        self.labels = labels
        return imgs, labels

    def augment(self):
        for i in range(len(self.imgs)):
            # Flip image vertically and invert angle
            self.imgs.append(self.imgs[i].with_transformation(lambda img: np.fliplr(img)))
            self.labels.append(-self.labels[i])

    def validation_split(self, percentage=0.2):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.imgs, self.labels,
                                                                                  test_size=percentage)
        return self.x_train, self.x_valid, self.y_train, self.y_valid

    @staticmethod
    def generate_set(x, y, batch_size):
        while 1:
            x, y = shuffle(x, y)
            for offset in range(0, len(x), batch_size):
                batch_x = np.array([image_def.load_image() for image_def in x[offset:offset + batch_size]])
                batch_y = np.array(y[offset:offset + batch_size])
                yield shuffle(batch_x, batch_y)

    def train_set(self, batch_size):
        return self.generate_set(self.x_train, self.y_train, batch_size)

    def valid_set(self, batch_size):
        return self.generate_set(self.x_valid, self.y_valid, batch_size)
