#Henry Zhong
from scipy import misc
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import matplotlib.pyplot as plt
import matplotlib as mp

from imageio import imread
from PIL import Image

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W = tf.truncated_normal(shape, stddev = 0.1)
    W = tf.Variable(W)
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.constant(0.1, shape = shape)
    b = tf.Variable(b)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    h_conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return h_max


ntrain = 500 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 50

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
base_data_path = '/content/drive/MyDrive/CIFAR10/'
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        sub_data_path = 'Train/%d/Image%05d.png' % (iclass,isample)
        path = base_data_path + sub_data_path
        # im = imread(path); # 28 by 28
        im = np.asarray(Image.open(path), dtype = np.float64)
        # print(f'Image {path} done')
        im = im.astype(float)/255
        # if isample%10 == 0:
        #     print(f'\r{isample+1}/{ntrain} done.\r')
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    print(f'Loaded {ntrain} train images from class {iclass}.')
    for isample in range(0, ntest):
        sub_data_path = 'Test/%d/Image%05d.png' % (iclass,isample)
        path = base_data_path + sub_data_path
        # im = imread(path); # 28 by 28
        im = np.asarray(Image.open(path), dtype = np.float64)
        im = im.astype(float)/255
        # if isample%10 == 0:
        #     print(f'\r{isample+1}/{ntest} done.\r')
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable
    print(f'Loaded {ntest} test images from class {iclass}.')

sess = tf.InteractiveSession()

#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, [None, imsize, imsize, nchannels])
#tf variable for labels
tf_labels = tf.placeholder(tf.float32, [None, nclass])

# --------------------------------------------------
# model
#create your model

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))

learning_rate = tf.placeholder(tf.float32, shape=[])

optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros([batchsize, imsize, imsize, nchannels]) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass]) #setup as [batchsize, the how many classes]

nsamples = ntrain * nclass

log_list = []
for i in range(2000): # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)

    it_feed_dict = {tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    it_loss = cross_entropy.eval(feed_dict = it_feed_dict)
    it_acc = accuracy.eval(feed_dict = it_feed_dict)
    # record params
    # first_weight = W_conv1.eval()
    if i%100 == 0:
        #calculate train accuracy and print it
        it_test_acc = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        print(f'Step #{i}. Train Loss: {it_loss}; Train Acc: {it_acc}; Test Acc: {it_test_acc}')
        log_list.append((i, it_loss, it_acc, it_test_acc))
    optimizer.run(feed_dict = it_feed_dict) # dropout only during training

# --------------------------------------------------
# test


print('#'*20)
print(log_list)
print("Final test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

# --------------------------------------------------
# visualization

first_conv_weights = W_conv1.eval()


fig = plt.figure()
for i in range(32):
    ax = fig.add_subplot(4, 8, 1 + i)
    ax.imshow(first_conv_weights[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.savefig('1st conv layer')
plt.show()

# --------------------------------------------------
# activation stats

h_conv1_stats = np.array(h_conv1.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
h_conv2_stats = np.array(h_conv2.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
print(f'h_conv1_stats: mean = {np.mean(h_conv1_stats)}; var = {np.var(h_conv1_stats)}')
print(f'h_conv2_stats: mean = {np.mean(h_conv2_stats)}; var = {np.var(h_conv2_stats)}')


sess.close()