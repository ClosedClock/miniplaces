import os, datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from DataLoader import *

from resnet import resnet18


# Dataset Parameters
batch_size = 100
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097, 0.44674252445, 0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5  # Dropout, probability to keep units
training_iters = 100000
step_display = 10
step_save = 2000
path_save = 'resnet.pt'
start_from = ''
test_result_file = 'test_prediction.txt'

do_training = True
do_validation = True
do_testing = False

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',  # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt',  # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
}
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',  # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',  # MODIFY PATH ACCORDINGLY
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
# loader_train = DataLoaderH5(**opt_data_train)
# loader_val = DataLoaderH5(**opt_data_val)

net = resnet18(pretrained=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def accuracy(k, outputs_batch, labels_batch):
    predictions_batch = outputs_batch.topk(k, dim=1)[1]
    result = torch.sum(predictions_batch == labels_batch.unsqueeze(1)).double() / outputs_batch.size()[0]
    # print('calculating accuracy %d' % k)
    # print(predictions_batch)
    # print(labels_batch.unsqueeze(1))
    # print(predictions_batch == labels_batch.unsqueeze(1))
    # print(result)
    return result.data[0]


def top5_labels(outputs_batch):
    return outputs_batch.topk(5, dim=1)[1]

# Launch the graph

if do_training:
    if len(start_from)>1:
        net.load_state_dict(torch.load(start_from))
        print('Started from last time: %s' % start_from)

    for step in range(training_iters):
        # zero the parameter gradients
        optimizer.zero_grad()  # 将参数的grad值初始化为0

        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        images_batch, labels_batch = Variable(images_batch), Variable(labels_batch)

        outputs = net(images_batch)
        loss = criterion(outputs, labels_batch)

        if step % step_display == 0:
            print('[%s]:' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set

            acc1 = accuracy(1, outputs, labels_batch)
            acc5 = accuracy(5, outputs, labels_batch)

            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.4f}".format(loss.data[0]) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            images_batch_val, labels_batch_val = Variable(images_batch_val), Variable(labels_batch_val)
            outputs_val = net(images_batch_val)
            loss = criterion(outputs_val, labels_batch_val)  # 将output和labels使用叉熵计算损失
            acc1 = accuracy(1, outputs_val, labels_batch_val)
            acc5 = accuracy(5, outputs_val, labels_batch_val)
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.4f}".format(loss.data[0]) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

        # forward + backward + optimize
        loss.backward()  # 反向传播
        optimizer.step()  # 用SGD更新参数

        # Save model
        if step % step_save == 0:
            torch.save(net.state_dict(), path_save)
            print("Model saved at Iter %d !" % (step))

    print("Optimization Finished!")


if do_validation:
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size() // batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch // 10):
        images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
        images_batch_val, labels_batch_val = Variable(images_batch_val), Variable(labels_batch_val)
        outputs_val = net(images_batch_val)
        acc1 = accuracy(1, outputs_val, labels_batch_val)
        acc5 = accuracy(5, outputs_val, labels_batch_val)
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(
        acc5_total))


if do_testing:
    # Test on the test set
    print('Evaluation on the whole test set...')
    num_batch = loader_test.size()//batch_size
    loader_test.reset()

    with open(test_result_file, 'w') as f:
        print('Opened file %s' % test_result_file)
        for i in range(num_batch):
            print('There are %d test images left' % (loader_test.size() - i * batch_size))
            images_batch_test, filenames_batch = loader_test.next_batch(batch_size)
            images_batch_test = Variable(images_batch_test)
            outputs_test = net(images_batch_test)
            predicted_labels = top5_labels(outputs_test)
            for j in range(len(filenames_batch)):
                f.write(filenames_batch[j] + ' %d %d %d %d %d\n' % tuple(predicted_labels.data[j, :].tolist()))

    print('Test Finished!')