import torch
import torch.distributed as dist
from utils import get_sample_writer
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed

import resnet_model


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results.")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def master_process(args):
    # return (not args.no_cuda
    #         and dist.get_rank() == 0) or (args.no_cuda
    #                                       and args.local_rank == -1)
    return dist.get_rank() == 0


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

args = add_argument()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1024,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# # functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net()
net = resnet_model.PreActResNet18()
# parameters = filter(lambda p: p.requires_grad, net.parameters())
parameters, names = [], []
for n, p in list(net.named_parameters()):
    if p.requires_grad:
        parameters.append(p)
        names.append(n)
print('num parameter: ', len(names), len(parameters))
print('parameter names: ', names)
optimizer_grouped_parameters = [{'params': parameters, 'no_freeze': False}]

# parameters, names = [], []
# parameters2, names2 = [], []
# for n, p in list(net.named_parameters()):
#     if p.requires_grad:
#         if n[0:6] == 'linear':
#             parameters2.append(p)
#             names2.append(n)
#         else:
#             parameters.append(p)
#             names.append(n)
# optimizer_grouped_parameters = [{'params': parameters, 'no_freeze': False}, {'params': parameters2, 'no_freeze': True}]
# names = names+names2

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=optimizer_grouped_parameters,
    training_data=trainset)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

if master_process(args):
    summary_writer = get_sample_writer(name=args.job_name, base='./output/')
global_step = 0
global_data_samples = 0

use_lamb = (model_engine.optimizer_name() ==
            deepspeed.pt.deepspeed_config.LAMB_OPTIMIZER
            or model_engine.optimizer_name() ==
            deepspeed.runtime.config.ONEBIT_LAMB_OPTIMIZER)


# Lamb config
def update_lr_this_step():
    global global_step
    lr_offset = 0
    warmup_proportion = 0.1
    learning_rate = 1e-2
    decay_rate = 0.9
    decay_step = 250
    total_training_steps = 10000
    degree = 2.0

    x = global_step / total_training_steps
    warmup_end = warmup_proportion * total_training_steps
    if x < warmup_proportion:
        # lr_this_step = (x / warmup_proportion)**degree
        lr_this_step = x / warmup_proportion
    else:
        # lr_this_step = decay_rate**((global_step - warmup_end) / decay_step)
        lr_this_step = 1
    lr_this_step = lr_this_step * learning_rate
    lr_this_step += lr_offset
    return lr_this_step


for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    model_engine.train()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        global_data_samples += (model_engine.train_micro_batch_size_per_gpu() *
                                dist.get_world_size())

        model_engine.backward(loss)

        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            if use_lamb:
                lr_this_step = update_lr_this_step()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            model_engine.step()
            # print statistics
            if master_process(args):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                summary_writer.add_scalar(f'Train/step/train_loss',
                                          loss.item(), global_step)
                summary_writer.add_scalar(f'Train/sample/train_loss',
                                          loss.item(), global_data_samples)
                if use_lamb:
                    summary_writer.add_scalar(f'Train/sample/lr', lr_this_step,
                                              global_data_samples)
                    lamb_coeffs = optimizer.get_lamb_coeffs()
                    if len(lamb_coeffs) > 0:
                        assert len(lamb_coeffs) == len(names)
                        for i in range(len(lamb_coeffs)):
                            summary_writer.add_scalar(
                                'Lamb/step/coeff_{}_{}'.format(i, names[i]),
                                lamb_coeffs[i], global_step)
                            # summary_writer.add_scalar('Lamb/sample/coeff_{}_{}'.format(i, names[i]), lamb_coeffs[i], global_data_samples)
        else:
            model_engine.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(
                model_engine.local_rank)).sum().item()
    if master_process(args):
        summary_writer.add_scalar(f'Test/epoch/test_accuracy',
                                  float(correct) / total, epoch + 1)
        summary_writer.add_scalar(f'Test/sample/test_accuracy',
                                  float(correct) / total, global_data_samples)

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
if master_process(args):
    print('GroundTruth: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images.to(model_engine.local_rank))

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

if master_process(args):
    print('Predicted: ',
          ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()
if master_process(args):
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

if master_process(args):
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))
