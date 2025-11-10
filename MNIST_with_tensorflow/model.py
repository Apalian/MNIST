import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Model:
    def __init__(self):
        self.model = None
        self.history = None
        self.input_shape = (784,)
        self.num_classes = 10

    def build(self):
        if self.model is not None:
            return
        
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(784,)))
        self.model.add(Dense(10, activation='softmax'))

    def train(self, epochs=5, batch_size=32):
        self.build()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train / 255.0
        x_train = x_train.reshape((-1, 784))
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        
        x_test = x_test / 255.0
        x_test = x_test.reshape((-1, 784))
        y_test = to_categorical(y_test, num_classes=10)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    def evaluate(self):
        (_, _), (x_test, y_test) = mnist.load_data()
        
        x_test = x_test / 255.0
        x_test = x_test.reshape((-1, 784))
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        return self.model.evaluate(x_test, y_test)

    def predict(self, image):
        if len(image.shape) == 1:
            image = image.reshape(1, 784)
        
        predictions = self.model.predict(image)
        return predictions.argmax(axis=1)
    
    def predict_proba(self, image):
        if len(image.shape) == 1:
            image = image.reshape(1, 784)
        
        return self.model.predict(image)


    def save(self):
        self.model.save('model.h5')

    @classmethod
    def load(cls, filepath='model.h5'):
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance