import numpy as np

from nnc import NeuralNetworkClassifier
from fully_conn_layer import FCLayer
from activation_layer import ActivationLayer
from activation_funcs import tanh, tanh_prime, sigmoid, sigmoid_prime
from loss import mse, mse_prime

# xor training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


# (self, hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum)
net = NeuralNetworkClassifier(epochs = 1000, hidden_layer_sizes = 0, batch_size=0, learning_rate=0.1, max_iter=0, random_state=0, momentum=0.5)
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))

net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))



net.use(mse, mse_prime)
net.fit(x_train, y_train)


out = net.predict(x_train)
print(out)
