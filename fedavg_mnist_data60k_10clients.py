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
cls_train = np.matmul(labels_train, np.arange(10))

# Create non-IID data
n_shards = 20
order = cls_train.argsort()
cls_train_sorted = cls_train[order]
images_train_sorted = images_train[order]
labels_train_sorted = labels_train[order]
images_train_split = np.vsplit(images_train_sorted, n_shards)
labels_train_split = np.vsplit(labels_train_sorted, n_shards)

from random import shuffle, seed
seed(101)
shuffle(images_train_split)
seed(101)
shuffle(labels_train_split)

images_train_split_nonIID = []
labels_train_split_nonIID = []
while images_train_split:
    images_train_split_nonIID.append(np.vstack((images_train_split.pop(), images_train_split.pop())))
    labels_train_split_nonIID.append(np.vstack((labels_train_split.pop(), labels_train_split.pop())))

for i in range(len(images_train_split_nonIID)):
    np.random.seed(i)
    np.random.shuffle(images_train_split_nonIID[i])
    np.random.seed(i)
    np.random.shuffle(labels_train_split_nonIID[i])

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

pw1 = tf.placeholder(tf.float32, [5, 5, 1, 32])
opw1 = tf.assign(W_conv1, pw1)

pb1 = tf.placeholder(tf.float32, [32])
opb1 = tf.assign(b_conv1, pb1)

pw2 = tf.placeholder(tf.float32, [5, 5, 32, 64])
opw2 = tf.assign(W_conv2, pw2)

pb2 = tf.placeholder(tf.float32, [64])
opb2 = tf.assign(b_conv2, pb2)

pw3 = tf.placeholder(tf.float32, [7*7*64, 512])
opw3 = tf.assign(W_fc1, pw3)

pb3 = tf.placeholder(tf.float32, [512])
opb3 = tf.assign(b_fc1, pb3)

pw4 = tf.placeholder(tf.float32, [512, 10])
opw4 = tf.assign(W_fc2, pw4)

pb4 = tf.placeholder(tf.float32, [10])
opb4 = tf.assign(b_fc2, pb4)

#Training
rounds = 2
E = 1
batch_size = 100
C = 1.0
K = 10
init_lr = 0.01
decay = 0.995
learning_rate = init_lr
total_batch = int(images_train_split_nonIID[0].shape[0]*K/K/batch_size)
print(total_batch)

sess_nonIID = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
saver.restore(sess_nonIID, './sgd_mnist_paper/model.ckpt')

fedavg_nonIID_accuracy = []
fedavg_nonIID_loss = []
fedavg_nonIID_accuracy.append(sess_nonIID.run(accuracy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                                   y_: labels_test}))
fedavg_nonIID_loss.append(sess_nonIID.run(cross_entropy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                                   y_: labels_test}))
print('Round 0: accuracy={}'.format(fedavg_nonIID_accuracy[0]))
print('Round 0: loss={}'.format(fedavg_nonIID_loss[0]))


start_time = time.time()
for r in range(rounds):
    # randomly sielect m clients
    m = max(C*K, 1)
    s = np.random.choice(int(K), int(m), replace=False)
    
    # store shared weights and biases in the current round
    temp_weights = sess_nonIID.run([W_conv1, W_conv2, W_fc1, W_fc2])
    temp_biases = sess_nonIID.run([b_conv1, b_conv2, b_fc1, b_fc2])
    
    # empty weights and biases from clients
    clients_weights, clients_biases = [], []
    
    # use the same shared weights and biases for all clients in the current round
    for client in s:  
        for epoch in range(E):           
            start, end = 0, batch_size
            
            for i in range(total_batch):
                batch_x, batch_y = images_train_split_nonIID[client][start:end, :], labels_train_split_nonIID[client][start:end, :]
                start += batch_size
                end += batch_size
                sess_nonIID.run(train_step, feed_dict={x: batch_x.reshape(batch_size, 28, 28, 1), 
                                                       y_: batch_y, lr: learning_rate})

                    
        clients_weights.append(sess_nonIID.run([W_conv1, W_conv2, W_fc1, W_fc2]))    
        clients_biases.append(sess_nonIID.run([b_conv1, b_conv2, b_fc1, b_fc2]))
        
        # reset shared weights and biases
        # all clients use the parameter in the previous round
        sess_nonIID.run(opw1, feed_dict={pw1: temp_weights[0]})
        sess_nonIID.run(opw2, feed_dict={pw2: temp_weights[1]})
        sess_nonIID.run(opw3, feed_dict={pw3: temp_weights[2]})
        sess_nonIID.run(opw4, feed_dict={pw4: temp_weights[3]})
        #sess_nonIID.run(opw5, feed_dict={pw5: temp_weights[4]})
        sess_nonIID.run(opb1, feed_dict={pb1: temp_biases[0]})
        sess_nonIID.run(opb2, feed_dict={pb2: temp_biases[1]})
        sess_nonIID.run(opb3, feed_dict={pb3: temp_biases[2]})
        sess_nonIID.run(opb4, feed_dict={pb4: temp_biases[3]})
        #sess_nonIID.run(opb5, feed_dict={pb5: temp_biases[4]})
    
    # compute new federated averaging(fedavg) weights and biases
    fedavg_weights = fedavg(clients_weights)
    fedavg_biases = fedavg(clients_biases)
    
    # update shared weights and biases with fedavg
    sess_nonIID.run(opw1, feed_dict={pw1: fedavg_weights[0]})
    sess_nonIID.run(opw2, feed_dict={pw2: fedavg_weights[1]})
    sess_nonIID.run(opw3, feed_dict={pw3: fedavg_weights[2]})
    sess_nonIID.run(opw4, feed_dict={pw4: fedavg_weights[3]})
    #sess_nonIID.run(opw5, feed_dict={pw5: fedavg_weights[4]})
    sess_nonIID.run(opb1, feed_dict={pb1: fedavg_biases[0]})
    sess_nonIID.run(opb2, feed_dict={pb2: fedavg_biases[1]})
    sess_nonIID.run(opb3, feed_dict={pb3: fedavg_biases[2]})
    sess_nonIID.run(opb4, feed_dict={pb4: fedavg_biases[3]})
    #sess_nonIID.run(opb5, feed_dict={pb5: fedavg_biases[4]})
    
    test_accuracy = sess_nonIID.run(accuracy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                         y_: labels_test})
    test_loss = sess_nonIID.run(cross_entropy, feed_dict={x: images_test.reshape(10000, 28, 28, 1), 
                                                       y_: labels_test})
    fedavg_nonIID_accuracy.append(test_accuracy)
    fedavg_nonIID_loss.append(test_loss)
    
    if r % 1 == 0: 
        print('Epoch {0}: loss={1}, accuracy={2}, lr={3}'.format(r, test_loss, test_accuracy, learning_rate))
        
    learning_rate *= decay

end_time = time.time()
print('Time: %.2fs' % (end_time-start_time))

with open('./fedavg_mnist_data60k_10clients.pickle', 'wb') as handle:
    pickle.dump({'parameters': [batch_size, init_lr, decay],
                'fedavg_nonIID_accuracy': fedavg_nonIID_accuracy,
                'fedavg_nonIID_loss': fedavg_nonIID_loss},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)

sess_nonIID.close()

