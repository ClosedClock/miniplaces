import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *


# Dataset Parameters
batch_size = 100
# batch_size = 256
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
# training_iters = 50000
training_iters = 100000
do_training = False
do_validation = True
do_testing = False
step_display = 10
step_save = 5000
path_save = './alexnet_bn'
start_from_1 = 'trained_model/droping_learning_rate/alexnet_bn-15000'
start_from_2 = 'trained_model/one_more_layer/alexnet_bn-10000'
test_result_file = 'test_prediction.txt'

# # Start checking for rate reductions
# check_reduce_rate_threshold = 2000
# lowest_learning_rate = 0.000001
#
# # Iterations to check if average accuracy has increased
# check_reduce_rate = 1000 // step_display

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
                      updates_collections=None,
                      is_training=train_phase,
                      reuse=None,
                      trainable=True,
                      scope=scope_bn)


def alexnet1(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2. / (11 * 11 * 3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2. / (5 * 5 * 96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2. / (3 * 3 * 256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2. / (3 * 3 * 384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),

        'wf6': tf.Variable(tf.random_normal([7 * 7 * 256, 4096], stddev=np.sqrt(2. / (7 * 7 * 256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2. / 4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2. / 4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    # conv5 = conv4
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out



def alexnet2(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2. / (11 * 11 * 3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2. / (5 * 5 * 96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2. / (3 * 3 * 256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2. / (3 * 3 * 384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),
        'wc5-2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),

        'wf6': tf.Variable(tf.random_normal([7 * 7 * 256, 4096], stddev=np.sqrt(2. / (7 * 7 * 256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2. / 4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2. / 4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    # conv5 = conv4
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + Pool, 13->6
    conv5_2 = tf.nn.conv2d(conv5, weights['wc5-2'], strides=[1, 1, 1, 1], padding='SAME')
    conv5_2 = batch_norm_layer(conv5_2, train_phase, 'bn5-2')
    conv5_2 = tf.nn.relu(conv5_2)
    pool5_2 = tf.nn.max_pool(conv5_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5_2, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out








# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_test.h5',
    'data_root': '../../data/images/test/',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

if do_training:
    loader_train = DataLoaderDisk(**opt_data_train)
if do_validation:
    loader_val = DataLoaderDisk(**opt_data_val)
if do_testing:
    loader_test = TestDataLoaderDisk(**opt_data_test)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)


# tf Graph input
x1 = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y1 = tf.placeholder(tf.int64, None)
keep_dropout_1 = tf.placeholder(tf.float32)
train_phase_1 = tf.placeholder(tf.bool)

x2 = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y2 = tf.placeholder(tf.int64, None)
keep_dropout_2 = tf.placeholder(tf.float32)
train_phase_2 = tf.placeholder(tf.bool)

# Construct model
logits1 = alexnet1(x1, keep_dropout_1, train_phase_1)
logits2 = alexnet2(x2, keep_dropout_2, train_phase_2)
# top5_values, top5_labels = tf.nn.top_k(logits, k=5)
top10_values_1, top10_labels_1 = tf.nn.top_k(logits1, k=10)
top10_values_2, top10_labels_2 = tf.nn.top_k(logits2, k=10)


def combine_results(top10_labels_1, top10_labels_2):
    N =top10_labels_1.size()[0]
    result = np.zeros((N, 5), dtype=np.int)
    for i in range(N):
        same_labels = [label for label in top10_labels_1[i, :] if label in top10_labels_2[i, :]]
        index = 0
        while len(same_labels) < 5:
            if top10_labels_1[index] not in same_labels:
                same_labels.append(top10_labels_1[index])
            if top10_labels_2[index] not in same_labels:
                same_labels.append(top10_labels_2[index])
            index += 1
        result[i, :] = same_labels[0:5]
    return result


def accuracy(top10_labels_1, top10_labels_2, labels_batch, k):
    result = combine_results(top10_labels_1, top10_labels_2)
    N = top10_labels_1.size()[0]
    correct_num = 0
    for i in range(N):
        if labels_batch[i] in result[i, 0:k]:
            correct_num += 1
    return correct_num / N


# Define loss and optimizer
loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
train_optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss1)
train_optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2)

# Evaluate model
# accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
# accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver# Build a graph containing `net1`.

with tf.Graph().as_default() as net1_graph:
  net1 = alexnet1(x1, keep_dropout_1, train_phase_1)
  saver1 = tf.train.Saver(...)
sess1 = tf.Session(graph=net1_graph)
saver1.restore(sess1, start_from_1)

# Build a separate graph containing `net2`.
with tf.Graph().as_default() as net2_graph:
  net2 = alexnet2(x2, keep_dropout_2, train_phase_2)
  saver2 = tf.train.Saver(...)
sess2 = tf.Session(graph=net2_graph)
saver2.restore(sess2, start_from_2)

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# # Launch the graph
# with tf.Session() as sess1:
#     # Initialization
#     if len(start_from_1)>1:
#         saver1.restore(sess1, start_from_1)
#         print('Started from last time: %s' % start_from_1)
#     else:
#         sess1.run(init)
#         print('Initialized')
#
#     with tf.Session() as sess2:
#         # Initialization
#         if len(start_from_2)>1:
#             saver2.restore(sess2, start_from_2)
#             print('Started from last time: %s' % start_from_2)
#         else:
#             sess2.run(init)
#             print('Initialized')

        # Evaluate on the whole validation set
if do_validation:
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch//10):
        images_batch, labels_batch = loader_val.next_batch(batch_size)
        #acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})

        labels_1 = sess1.run(top10_labels_1, feed_dict={x1: images_batch, keep_dropout_1: 1., train_phase_1: False})
        # labels_2 = sess2.run(top10_labels_2, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
        labels_2 = 0
        predicted_labels = combine_results(labels_1, labels_2)

        acc1 = accuracy(labels_1, labels_2, labels_batch, 1)
        acc5 = accuracy(labels_1, labels_2, labels_batch, 1)
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))


if do_testing:
    # Test on the test set
    print('Evaluation on the whole test set...')
    num_batch = loader_test.size()//batch_size
    loader_test.reset()

    with open(test_result_file, 'w') as f:
        print('Opened file %s' % test_result_file)
        for i in range(num_batch):
            print('There are %d test images left' % (loader_test.size() - i * batch_size))
            images_batch, filenames_batch = loader_test.next_batch(batch_size)
            # predicted_labels.shape = (batch_size, 5)
            labels_1 = sess1.run(top10_labels_1, feed_dict={x1: images_batch, keep_dropout_1: 1., train_phase_1: False})
            #labels_2 = sess2.run(top10_labels_2, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})
            labels_2 = 0
            predicted_labels = combine_results(labels_1, labels_2)
            for j in range(len(filenames_batch)):
                f.write(filenames_batch[j] + ' %d %d %d %d %d\n' % tuple(predicted_labels[j, :]))

    print('Test Finished!')

