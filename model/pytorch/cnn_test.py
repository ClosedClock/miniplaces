import os, datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from DataLoader import *

# Dataset Parameters
batch_size = 10
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097, 0.44674252445, 0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5  # Dropout, probability to keep units
training_iters = 10
step_display = 2
step_save = 10000
path_save = 'alexnet'
start_from = ''


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            # self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1,
            #         padding=(int((local_size-1.0)/2), 0, 0))
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1)
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    # Difference with tensorflow model: padding is zero for each Conv2d and MaxPool2d,
    # 'SAME' for tensorflow model. Correctness of sizes of padding hasn't been checked
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        # I changed padding to 2 to get correct dimension
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv1.weight.data.normal_(0, 2/(11*11*3))
        # self.lrn1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv2.weight.data.normal_(0, 2/(5*5*96))
        # self.lrn = LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv3.weight.data.normal_(0, 2/(3*3*256))

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv4.weight.data.normal_(0, 2/(3*3*384))

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.conv5.weight.data.normal_(0, 2/(3*3*256))
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1.weight.data.normal_(0, 2/(256*6*6))
            # nn.Dropout(p=1 - keep_prob), # Need to be added later
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2.weight.data.normal_(0, 2/4096)
            # nn.Dropout(p=1 - keep_prob),
        self.out = nn.Linear(4096, num_classes)
        self.out.weight.data.normal_(0, 2/4096)

    def forward(self, x):
        # x = self.pool1(self.lrn1(F.relu(self.conv1(x))))
        # x = self.pool2(self.lrn2(F.relu(self.conv2(x))))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(x.size(0), 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


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

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
# loader_train = DataLoaderH5(**opt_data_train)
# loader_val = DataLoaderH5(**opt_data_val)

net = AlexNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def accuracy(k, outputs_batch, labels_batch):
    predictions_batch = outputs_batch.topk(k, dim=1)[1]
    result = torch.sum(predictions_batch == labels_batch.unsqueeze(1)) / outputs_batch.size()[0]
    return result.data[0]


# Launch the graph

for step in range(training_iters):
    # Load a batch of training data
    images_batch, labels_batch = loader_train.next_batch(batch_size)
    images_batch, labels_batch = Variable(images_batch), Variable(labels_batch)

    if step % step_display == 0:
        print('[%s]:' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # Calculate batch loss and accuracy on training set

        outputs = net(images_batch)
        loss = criterion(outputs, labels_batch)
        acc1 = accuracy(1, outputs, labels_batch)
        acc5 = accuracy(5, outputs, labels_batch)

        print("-Iter " + str(step) + ", Training Loss= " + \
              "{:.4f}".format(loss.data[0]) + ", Accuracy Top1 = " + \
              "{:.2f}".format(acc1) + ", Top5 = " + \
              "{:.2f}".format(acc5))

        # Calculate batch loss and accuracy on validation set
        images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
        images_batch_val = Variable(images_batch_val)
        labels_batch_val = Variable(labels_batch_val)
        outputs_val = net(images_batch_val)
        loss = criterion(outputs_val, labels_batch_val)  # 将output和labels使用叉熵计算损失
        acc1 = accuracy(1, outputs_val, labels_batch_val)
        acc5 = accuracy(5, outputs_val, labels_batch_val)
        print("-Iter " + str(step) + ", Validation Loss= " + \
              "{:.4f}".format(loss.data[0]) + ", Accuracy Top1 = " + \
              "{:.2f}".format(acc1) + ", Top5 = " + \
              "{:.2f}".format(acc5))


    # zero the parameter gradients
    optimizer.zero_grad()  # 将参数的grad值初始化为0

    # forward + backward + optimize
    outputs = net(images_batch)
    loss = criterion(outputs, labels_batch)  # 将output和labels使用叉熵计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 用SGD更新参数

    step += 1

    # Save model
    if step % step_save == 0:
        torch.save(net.state_dict(), path_save)
        print("Model saved at Iter %d !" % (step))

print("Optimization Finished!")

# Evaluate on the whole validation set
print('Evaluation on the whole validation set...')
num_batch = loader_val.size() // batch_size
acc1_total = 0.
acc5_total = 0.
loader_val.reset()
for i in range(num_batch):
    images_batch, labels_batch = loader_val.next_batch(batch_size)
    acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
    acc1_total += acc1
    acc5_total += acc5
    print("Validation Accuracy Top1 = " + \
          "{:.2f}".format(acc1) + ", Top5 = " + \
          "{:.2f}".format(acc5))

acc1_total /= num_batch
acc5_total /= num_batch
print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(
    acc5_total))
