from sklearn.utils import np_version
__author__ = 'henry_zhong'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import three_layer_neural_network

def generate_data(dataset):
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    if dataset == 'make_moons':
        X, y = datasets.make_moons(200, noise=0.20)
    elif dataset == 'make_circles':
        X, y = datasets.make_circles(200, noise=.20)
    elif dataset == 'make_gaussian_quantiles':
        X, y = datasets.make_gaussian_quantiles(n_features=2, n_classes=3)
    else:
        raise Exception(f'Input dataset <{dataset}> is not supported.')

    print(f'X.shape: {X.shape}')
    print(f'y.shape: {y.shape}')

    return X, y


class Layer(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda


        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE

        if type == 'Tanh':
            return np.tanh(z)
        elif type == 'Sigmoid':
            return 1.0/(1 + np.exp(-z))
        elif type == 'ReLU':
            return z * (z > 0)
        else:
            raise Exception(f'Input activation type, <{type}> is not supported')

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE

        if type == 'Tanh':
            return 1 - np.square(self.actFun(z, 'Tanh'))
        elif type == 'Sigmoid':
            return self.actFun(z, 'Sigmoid')*(1 - self.actFun(z, 'Sigmoid'))
        elif type == 'ReLU':
            return z > 0
        else:
            raise Exception(f'Input activation type, <{type}> is not supported')

        return None

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE

        self.X = X
        self.z = np.dot(self.X, self.W) + self.b
        self.a = self.actFun(self.z, type = self.actFun_type)

        return self.a


    def backprop(self, delta):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE

        delta = delta * (self.diff_actFun(self.z, type = self.actFun_type))
        self.dW = np.dot(self.X.T, delta) + self.reg_lambda * self.W
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)



class DeepNeuralNetwork(three_layer_neural_network.NeuralNetwork):
    def __init__(self, nn_layer_num, nn_layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):

        self.nn_layer_num = nn_layer_num
        self.nn_layer_sizes = nn_layer_sizes
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        self.nn_layers = []
        for i in range(self.nn_layer_num - 1):
            layer_nn_input_dim = self.nn_layer_sizes[i]
            layer_nn_output_dim  = self.nn_layer_sizes[i+1]
            layer = Layer(layer_nn_input_dim, layer_nn_output_dim, self.actFun_type)
            self.nn_layers.append(layer)


    def feedforward(self, X, actFun = None):
        """
        :param x: input
        :param actFun: the activation function passed as an anonymous function
        :return: probabilities
        """

        layer_a = X
        for a_layer in self.nn_layers:
            a_layer.feedforward(layer_a)
            layer_a = a_layer.a
        self.probs = layer_a


        if self.actFun_type == "ReLU": # avoid division by zero.
            self.probs = np.exp(layer_a) / np.sum(np.exp(layer_a), axis = 1, keepdims = True)


        return self.probs



    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE

        delta = self.probs
        examples = len(X)
        delta[range(examples), y] -= 1

        for a_layer in reversed(self.nn_layers):
            delta = a_layer.backprop(delta)

        return delta


    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)
        reg_sum = 0
        for a_layer in self.nn_layers:
            reg_sum += np.sum(np.square(a_layer.W))
        data_loss += (self.reg_lambda / 2) * reg_sum

        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            for a_layer in self.nn_layers:
                a_layer.dW += self.reg_lambda * a_layer.W

            # Gradient descent parameter update
            for a_layer in self.nn_layers:
                a_layer.W += -epsilon * a_layer.dW
                a_layer.b += -epsilon * a_layer.db

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


def main():
    X, y = generate_data('make_gaussian_quantiles')
    # X, y = generate_data('make_moons')
    # X, y = generate_data('make_circles')

    # layer_sizes = [X.shape[1], 50, 50, 50, 50, 50, 3]
    # # layer_sizes = [X.shape[1], 15, 5, 3, 2]
    # # layer_sizes = [X.shape[1], 3, 3, 3, 2]
    layer_sizes = [X.shape[1], 20, 20, 20, 20, 3]

    model = DeepNeuralNetwork(len(layer_sizes), layer_sizes, actFun_type='ReLU')

    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()