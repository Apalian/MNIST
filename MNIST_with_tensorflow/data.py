import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model import Model

ml = Model()
ml.train()

plt.plot(ml.history.history['loss'], label='loss')
plt.plot(ml.history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape((-1, 784))
y_test = to_categorical(y_test, num_classes=10)

y_pred = ml.predict(x_test)
confusion_mtx = tf.math.confusion_matrix(y_test.argmax(axis=1), y_pred)

ConfusionMatrixDisplay(confusion_mtx.numpy()).plot(ax=plt.gca())
plt.show()


