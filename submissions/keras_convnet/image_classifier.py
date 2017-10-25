import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam


class ImageClassifier(object):

    def __init__(self):
        inp = Input((28, 28, 1))
        # Block 1
        x = Conv2D(
            32, (3, 3), activation='relu', padding='same',
            name='block1_conv1')(inp)
        x = Conv2D(
            32, (3, 3), activation='relu', padding='same',
            name='block1_conv2')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block1_pool')(x)
        # dense
        x = Flatten(
            name='flatten')(x)
        x = Dense(
            512, activation='relu',
            name='fc1')(x)
        out = Dense(10, activation='softmax', name='predictions')(x)
        self.model = Model(inp, out)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])

    def _transform(self, x):
        # adding channel dimension at the last position
        x = np.expand_dims(x, axis=-1)
        # bringing input between 0 and 1
        x = x / 255.
        return x

    def fit(self, img_loader):
        # load the full data into memory
        nb = len(img_loader)
        # make a 4D tensor:
        # number of images x width x height x number of channels
        X = np.zeros((nb, 28, 28, 1))
        # one-hot encoding of the labels to set NN target
        Y = np.zeros((nb, 10))
        for i in range(nb):
            x, y = img_loader.load(i)
            X[i] = self._transform(x)
            # since labels are [0, ..., 9], label is the same as label index
            Y[i, y] = 1
        self.model.fit(X, Y, batch_size=32, validation_split=0.1, epochs=1)

    def predict_proba(self, img_loader):
        nb = len(img_loader)
        X = np.zeros((nb, 28, 28, 1))
        for i in range(nb):
            X[i] = self._transform(img_loader.load(i))
        return self.model.predict(X)
