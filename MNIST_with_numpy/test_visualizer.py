from neural_network import NeuralNetwork
from visualizer import NetworkVisualizer

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape(60000, 784).T / 255.0
Y_train = to_categorical(y_train, 10).T

X_test = x_test.reshape(10000, 784).T / 255.0
Y_test = to_categorical(y_test, 10).T

nn = NeuralNetwork(784, 128, 10, 0.01)

viz = NetworkVisualizer(nn)

nn.train(X_train, Y_train, X_test, Y_test, 
         epochs=20, batch_size=32)

# Visualisations
viz = NetworkVisualizer(nn)
viz.visualize_w1()
viz.visualize_w2_composite()


# nn.train(X_train, Y_train, X_test, Y_test, 
#          epochs=10, batch_size=32, visualizer=viz)
