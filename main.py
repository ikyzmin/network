import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import sklearn.cluster as skl
from random import *
import math
import os


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [np.random.uniform() for i in range(n_inputs + 1)]} for j in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [np.random.uniform() for i in range(n_hidden + 1)]} for j in range(n_outputs)]
    network.append(output_layer)
    return network


def initialize_network_with_layers(layers):
    network = list()
    for i in range(len(layers) - 1):
        layer = [{'weights': [np.random.uniform() for j in range(layers[i] + 1)]} for k in range(layers[i + 1])]
        network.append(layer)
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


def update_weights_momentum(network, row, l_rate, momentum):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                if 'prevDelta' not in neuron:
                    neuron['prevDelta'] = 0
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] + momentum * neuron['prevDelta']
                neuron['prevDelta'] = neuron['delta']
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
        if epoch % 100 == 0:
            print('>epoch=%d, lrate=%.2f, error=%.5f' % (epoch, l_rate, sum_error))
        costs.append(sum_error)
    return costs


def train_network_with_momentum(network, train, output, l_rate, n_epoch, n_outputs):
    costs = list()
    momentum = 0.01
    for epoch in range(n_epoch):
        sum_error = 0
        rowIndex = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = output[rowIndex].tolist()[0]
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights_momentum(network, row, l_rate, momentum)
            rowIndex += 1
        sum_error /= len(train)
        sum_error = np.sqrt(sum_error)
        if epoch % 100 == 0:
            print('>epoch=%d, lrate=%.2f, error=%.5f' % (epoch, l_rate, sum_error))
        costs.append(sum_error)
    return costs


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def test(testInput, realInput, testOutput, file):
    error = 0
    i = 0
    for row in testInput:
        prediction = predict(network, row)
        error += sum(
            [(testOutput[i].tolist()[0][j] - prediction[j]) ** 2 for j in range(len(np.asarray(testOutput)[0]))])
        plt.plot(list(realInput[i].tolist()[0]), testOutput[i].tolist()[0], 'go', markersize=3, label="Контрольная выборка")
        plt.plot(list(realInput[i].tolist()[0]), prediction, 'rs', markersize=5, label="Данные персептрона")
        file.write(
            "Ожидаемый :\n\r" + str(testOutput[i].tolist()[0]) + "\n\rПолученный :\n\r" + str(prediction) + "\n\r")
        i += 1
    error /= len(testInput)
    plt.legend()
    plt.show()
    file.close()
    return np.sqrt(error)


seed(1)
data = open("data.txt")
dataOutputFuzzy = open("fuzzy_results.txt", 'w+')
dataOutputMLP = open("MLP_results.txt", 'w+')
dataOutputFuzzyMomentum = open("fuzzy_momentum_results.txt", 'w+')
dataOutputMLPMomentum = open("MLP_momentum_results.txt", 'w+')
dataOutputKmeansMomentum = open("kmeans_momentum_results.txt", 'w+')
dataOutputKmeans = open("kmeans_results.txt", 'w+')
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
lamb = 0.5
cost = 1
alf = 0.002
epochs = 800
c_means_epochs = 1500
npInput = np.asmatrix(inputRaw)
npOutput = np.asmatrix(outputRaw)
iteration = 0
fuzzyCost = list()
perceptronCost = list()
deltaAv = list()
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    npInput.T, 15, 2, error=0.002, maxiter=c_means_epochs, init=None)
fuzzyInput = u.T
n_inputs = len(fuzzyInput[0])
n_outputs = len(np.asarray(npOutput)[0])
clusters = list()
for row in npInput:
    clusters.append(skl.KMeans(n_clusters=15, random_state=0).fit_predict(np.transpose(row)))
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network_with_momentum(network, clusters[0:164], npOutput[0:164], lamb, epochs, n_outputs)
errorKmeansMomentum = test(clusters[164:206], npInput[164:206], npOutput[164:206], dataOutputKmeansMomentum)
print('k-means momentum test error = %.5f' % errorKmeansMomentum)
clusters = list()
for row in npInput:
    clusters.append(skl.KMeans(n_clusters=15, random_state=0).fit_predict(np.transpose(row), 0))
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network(network, clusters[0:164], npOutput[0:164], lamb, epochs, n_outputs)
errorKmeans =test(clusters[164:206], npInput[164:206], npOutput[164:206], dataOutputKmeans)
print('k-means test error = %.5f' % errorKmeans)
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network(network, fuzzyInput[0:164], npOutput[0:164], lamb, epochs, n_outputs)
errorFuzzy = test(fuzzyInput[164:206], npInput[164:206], npOutput[164:206], dataOutputFuzzy)
print('Fuzzy test error = %.5f' % errorFuzzy)
n_inputs = len(np.asarray(npInput)[0])
n_outputs = len(np.asarray(npOutput)[0])
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network(network, npInput[0:164], npOutput[0:164], lamb, epochs, n_outputs)
error = test(npInput[164:206], npInput[164:206], npOutput[164:206], dataOutputMLP)
print('MLP test error = %.5f' % error)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    npInput.T, 15, 2, error=0.002, maxiter=c_means_epochs, init=None)
fuzzyInput = u.T
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network_with_momentum(network, fuzzyInput[0:164], npOutput[0:164], lamb, epochs, n_outputs)
errorFuzzyMomentum = test(fuzzyInput[164:206], npInput[164:206], npOutput[164:206], dataOutputFuzzyMomentum)
print('Fuzzy momentum test error = %.5f' % errorFuzzyMomentum)
network = initialize_network_with_layers([n_inputs, 15, n_outputs])
train_network_with_momentum(network, fuzzyInput[0:164], npOutput[0:164], lamb, epochs, n_outputs)
errorMomentum = test(npInput[164:206], npInput[164:206], npOutput[164:206], dataOutputMLPMomentum)
print('MLP Momentum  test error = %.5f' % errorMomentum)
bar_width = 0.25
plt.bar(0, errorFuzzy, bar_width, label='c-means')
plt.bar(3 * bar_width / 2, errorFuzzyMomentum, bar_width, label='c-means momentum')
plt.bar(3 * bar_width, error, bar_width, label='perceptron')
plt.bar(9 * bar_width / 2, errorMomentum, bar_width, label='perceptron momentum')
plt.bar(6 * bar_width, errorKmeans, bar_width, label='k-means momentum')
plt.bar(15 * bar_width / 2, errorKmeansMomentum, bar_width, label='k-means')

plt.ylabel('Погрешность')
plt.xlabel('Персептрон')
plt.title('Погрешность тестирования сетей')
plt.xticks((0, 3 * bar_width / 2, 3 * bar_width, 9 * bar_width / 2, 6 * bar_width, 15 * bar_width / 2),
           ('c-means', 'c-means momentum', 'perceptron', "perceptron momentum", "k-means momentum", "k-means"))
plt.legend()
plt.tight_layout()
plt.show()
data.close()
