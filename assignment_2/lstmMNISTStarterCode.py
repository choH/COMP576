#Henry Zhong
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

# import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
# from tensorflow.python.ops import rnn_cell
import numpy as np

import assignment_2.input_data as input_data
# from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)#call mnist function

learningRate = 1e-3
trainingIters = 50000
batchSize = 128
displayStep = 20

nInput =  28#we want the input to take the 28 pixels
nSteps =  28 #every 28
nHidden = 64 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

	# lstmCell = rnn.BasicRNNCell(nHidden) #find which lstm to use in the documentation
	# lstmCell = rnn_cell.BasicRNNCell(nHidden) #find which lstm to use in the documentation
	# lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
	lstmCell = rnn_cell.GRUCell(nHidden)

	outputs, states = rnn.static_rnn(lstmCell, x, dtype=tf.float32)#for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

log_list = []
testData = mnist.test.images.reshape((-1, nSteps, nInput))
testLabel = mnist.test.labels
test_feed_dict = {x: testData, y: testLabel}
with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))


		batch_feed_dict = {x: batchX, y: batchY}
		sess.run(optimizer, feed_dict = batch_feed_dict)
		train_acc = sess.run(accuracy, feed_dict = batch_feed_dict)
		loss = sess.run(cost, feed_dict = batch_feed_dict)
		test_acc = sess.run(accuracy, feed_dict = test_feed_dict)
		if step % displayStep == 0:
			log_list.append([step * batchSize, loss, train_acc, test_acc])
			print("Iter " + str(step*batchSize) + ", Minibatch Loss = " + \
					  "{:.6f}".format(loss) + ", Training Accuracy = " + \
					  "{:.5f}".format(train_acc))
			print(f'Test acc: {test_acc}')


		step +=1

	print('Optimization finished')

	# testData = mnist.test.images.reshape((-1, nSteps, nInput))
	# testLabel = mnist.test.labels
	test_feed_dict = {x: testData, y: testLabel}
	print("Final Test Accuracy:", \
		sess.run(accuracy, feed_dict = test_feed_dict))

print(log_list)