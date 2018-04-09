import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

def sigma(x):
    return tf.div(tf.constant(1.0, dtype=tf.float64),
                  tf.add(tf.constant(1.0, dtype=tf.float64), tf.exp(tf.negative(x))))


data = open("data.txt")
raw_data = data.read().split("\n")
inputList = []
outputList = []
raw_data = list(filter(None, raw_data))
for i in range(len(raw_data)):
    if i % 2 != 0:
        inputList.append(list(map(float, raw_data[i].split())))
    else:
        outputList.append(list(map(float, raw_data[i].split())))
inputRaw = np.asarray(inputList)[0:164]
outputRaw = np.asarray(outputList)[0:164]

inputTest = np.asarray(inputList)[165:206]
outputTest = np.asarray(outputList)[165:206]
lr = 0.5
epochs = 800
batch_size = 100

x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)

testX = tf.placeholder(tf.float64)
testY = tf.placeholder(tf.float64)

hidden_neurons = 15
w1 = tf.Variable(tf.random_uniform(shape=[15, hidden_neurons], dtype=tf.float64), dtype=tf.float64)
b1 = tf.Variable(tf.constant(value=0.0, shape=[hidden_neurons], dtype=tf.float64))
layer1 = sigma(tf.add(tf.matmul(inputRaw, w1), b1))
w2 = tf.Variable(tf.random_uniform(shape=[hidden_neurons, 15], dtype=tf.float64), dtype=tf.float64)
b2 = tf.Variable(tf.constant(value=0.0, shape=[15], dtype=tf.float64))
nn_output = sigma(tf.add(tf.matmul(layer1, w2), b2))

loss = tf.reduce_mean(tf.square(nn_output - y))
testLoss = tf.reduce_mean(tf.square(nn_output[0:41] - y))
train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

trainCost = list()
fuzzyCost = list()

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
    inputRaw.T, 15, 2, error=0.002, maxiter=1500, init=None)

cntr, uTest, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    inputTest.T, 15, 2, error=0.002, maxiter=1500, init=None)

fuzzyInput = u.T
fuzzyInputTest = uTest.T

for epoch in range(10000):
    sess.run(train_step, feed_dict={x: fuzzyInput, y: outputRaw})
    c = sess.run(loss, feed_dict={x: fuzzyInput, y: outputRaw})
    fuzzyCost.append(c)
    if epoch % 1000 == 0:
        print("Epoch:", '%3d' % epoch, "fuzzy cost=", "{:9f}".format(c))


plt.plot(fuzzyInput, outputRaw, 'ro', markersize=0.5)
plt.plot(fuzzyInput, sess.run(nn_output, feed_dict={x: fuzzyInput}), 'go', markersize=0.5)
plt.show()

testCost = sess.run(testLoss, feed_dict={x: inputTest, y: outputTest})
print("fuzzy cost=", "{:9f}".format(testCost))
plt.title("Сопоставление данных Fuzzy C-means (Tenserflow)")
plt.plot(fuzzyInputTest, outputTest, 'bo', markersize=3)
plt.plot(fuzzyInputTest, sess.run(nn_output[0:41], feed_dict={x: fuzzyInputTest}), 'ro', markersize=3)
plt.legend()
plt.show()



for epoch in range(10000):
    sess.run(train_step, feed_dict={x: inputRaw, y: outputRaw})
    c = sess.run(loss, feed_dict={x: inputRaw, y: outputRaw})
    trainCost.append(c)
    if epoch % 1000 == 0:
        print("Epoch:", '%3d' % epoch, "cost=", "{:9f}".format(c))
plt.title("Срвнение погрешности обучения (красный - Fuzzy) Tenserflow")
plt.plot(fuzzyCost,'r')
plt.plot(trainCost,'b')
plt.ylim([0,0.0004])
plt.show()

plt.plot(inputRaw, outputRaw, 'ro', markersize=0.5)
plt.plot(inputRaw, sess.run(nn_output, feed_dict={x: inputRaw}), 'go', markersize=3)
plt.show()

testCost = sess.run(testLoss, feed_dict={x: inputTest, y: outputTest})
print("cost=", "{:9f}".format(testCost))
plt.title("Сопоставление данных Персептрон (Tenserflow)")
plt.plot(inputTest, outputTest, 'bo', markersize=0.5)
plt.plot(inputTest, sess.run(nn_output[0:41], feed_dict={x: inputTest}), 'ro', markersize=3)
plt.legend()
plt.show()
