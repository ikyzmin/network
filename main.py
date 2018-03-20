import copy
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from random import *
import math


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    inputsList = np.asmatrix(inputs).tolist()[0]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputsList[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, output, l_rate, n_epoch, n_outputs):
    costs = list()
    for epoch in range(n_epoch):
        sum_error = 0
        rowIndex = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = output[rowIndex].tolist()[0]
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            rowIndex += 1
        print('>epoch=%d, lrate=%.3f, error=%.10f' % (epoch, l_rate, sum_error))
        costs.append(sum_error)
    return costs


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs


seed(1)
data = open("data.txt")
raw_data = data.read().split("\n")
inputRaw = []
outputRaw = []
raw_data = list(filter(None, raw_data))
for i in range(len(raw_data)):
    if i % 2 != 0:
        inputRaw.append(list(map(float, raw_data[i].split())))
    else:
        outputRaw.append(list(map(float, raw_data[i].split())))

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
lamb = 1.850
cost = 1
alf = 0.002
npInput = np.asmatrix(inputRaw)
npOutput = np.asmatrix(outputRaw)
iteration = 0
fuzzyCost = list()
perceptronCost = list()
deltaAv = list()
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    npInput.T, 15, 2, error=0.002, maxiter=1500, init=None)

fuzzyInput = u.T
n_inputs = len(fuzzyInput[0])
n_outputs = len(np.asarray(npOutput)[0])
network = initialize_network(n_inputs, 15, n_outputs)
train_network(network, fuzzyInput[0:195], npOutput[0:195], lamb, 800, n_outputs)
i = 196
error = 0
testErrorFuzzy = list()
for row in fuzzyInput[196:206]:
    prediction = predict(network, row)
    error = sum([(npOutput[i].tolist()[0][j] - prediction[j]) ** 2 for j in range(len(np.asarray(npOutput)[0]))])
    print('Test error = ', error)
    testErrorFuzzy.append(error)
    i += 1
plt.plot(testErrorFuzzy, 'r')

n_inputs = len(np.asarray(npInput)[0])
n_outputs = len(np.asarray(npOutput)[0])
network = initialize_network(n_inputs, 15, n_outputs)
train_network(network, npInput[0:195], npOutput[0:195], lamb, 800, n_outputs)
i = 196
testError = list()
for row in npInput[196:206]:
    prediction = predict(network, row)
    error = sum([(npOutput[i].tolist()[0][j] - prediction[j]) ** 2 for j in range(len(np.asarray(npOutput)[0]))])
    print('Test error = ', error)
    testError.append(error)
    i += 1
plt.plot(testError, 'g')
plt.ylim([0, 0.0002])
plt.show()
data.close()
