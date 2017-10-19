import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


class ImageClassifier(object):

    def __init__(self):
        self.model = self._build_model()

    def transform(self, x):
        x = np.expand_dims(x, axis=-1)
        x = x / 255.
        return x

    def fit(self, img_loader):
        # load the full data into memory
        nb = len(img_loader)
        X = np.zeros((nb, 28, 28, 1))
        Y = np.zeros((nb, 10))
        for i in range(nb):
            x, y = img_loader.load(i)
            X[i] = self.transform(x)
            Y[i, y] = 1
        
        self.model.fit(X, Y, batch_size=32, validation_split=0.1, epochs=1)

    def predict_proba(self, test_img_loader):
        # load the full data into memory
        nb = len(test_img_loader)
        X = np.zeros((nb, 28, 28, 1))
        for i in range(nb):
            X[i] = self.transform(test_img_loader.load(i))
        return self.model.predict(X)

    def _build_model(self):
        inp = Input((28, 28, 1))
        x = Flatten(name='flatten')(inp)
        x = Dense(100, activation='relu', name='fc1')(x)
        out = Dense(10, activation='softmax', name='predictions')(x)
        model = Model(inp, out)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4),
            metrics=['accuracy'])
        return model
