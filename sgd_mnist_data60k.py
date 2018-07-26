import tensorflow as tf
import numpy as np
import time
import pickle
from functions import *

#Load data
#images_train, cls_train, labels_train = cifar10.load_training_data()
#images_test, cls_test, labels_test = cifar10.load_test_data()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images_test = mnist.test.images
labels_test = mnist.test.labels

images_train = mnist.train.images
labels_train = mnist.train.labels
images_val = mnist.validation.images
labels_val = mnist.validation.labels

images_train = np.vstack([images_train, images_val])
labels_train = np.vstack([labels_train, labels_val])

################################################
##########     CNN architecture          #######
################################################
img_size, num_channels, num_classes = 28, 1, 10
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')

def weight_variable(shape, stddev, wd):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, val):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32], stddev=0.1, wd=None)
b_conv1 = bias_variable([32], 0.1)
conv1 = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64], stddev=0.1, wd=None)
b_conv2 = bias_variable([64], 0.1)
conv2 = tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

W_fc1 = weight_variable([7 * 7 * 64, 512], stddev=0.1, wd=None)
b_fc1 = bias_variable([512], 0.1)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([512, 10], stddev=0.1, wd=None)
b_fc2 = bias_variable([10], 0.1)
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
lr = tf.placeholder(tf.float32, shape=[])
opt = tf.train.GradientDescentOptimizer(lr)
train_step = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Training
rounds=100
batch_size =100
init_lr = 0.01
total_batch_sgd = int(images_train.shape[0]/batch_size)
print(total_batch_sgd)

sess_sgd = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
saver.restore(sess_sgd, './sgd_mnist_paper/model.ckpt')

learning_rate = init_lr
decay = 0.995

sgd_accuracy = []
sgd_loss = []

sgd_accuracy.append(sess_sgd.run(accuracy, feed_dict={x: images_test.reshape(10000, 28, 28, 1),y_: labels_test}))
sgd_loss.append(sess_sgd.run(cross_entropy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), y_: labels_test}))
print(sgd_accuracy)
print(sgd_loss)

start_time = time.time()
for epoch in range(rounds):
    start, end = 0, batch_size
        
    for i in range(total_batch_sgd):
        batch_x, batch_y = images_train[start: end], labels_train[start: end]
        start += batch_size
        end += batch_size
        sess_sgd.run(train_step, feed_dict={x: batch_x.reshape(batch_size, 28, 28, 1), y_: batch_y, lr: learning_rate})
        
    test_accuracy = sess_sgd.run(accuracy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                      y_: labels_test})
    test_loss = sess_sgd.run(cross_entropy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                       y_: labels_test})
    sgd_accuracy.append(test_accuracy) 
    sgd_loss.append(test_loss)
    
    print('Epoch {0}: loss={1}, accuracy={2}, lr={3}'.format(epoch, test_loss, test_accuracy, learning_rate))
    learning_rate *= decay
end_time = time.time()
print("Training takes {}s".format(end_time-start_time))    

with open('./sgd_mnist_data60k.pickle', 'wb') as handle:
    pickle.dump({'parameters': [batch_size, init_lr, decay],
                'fedavg_nonIID_accuracy': sgd_accuracy,
                'fedavg_nonIID_loss': sgd_loss},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

sess_sgd.close()
