import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import time
import numpy as np
import calMetric as m


def store_model(model, filename):
    pickle_file = open(filename, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()


def load_model(filename):
    pickle_file = open(filename, 'rb')
    loaded = pickle.load(pickle_file)
    pickle_file.close()

    return loaded


class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        num_classes = pred.size()

        weight = pred.new_ones(num_classes) * self.smoothing / (pred.size(-1) - 1.)
        target = target.type(torch.int64)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        return (-weight * log_prob).sum(dim=-1).mean()


def train():
    densenet = torch.load("../models/{}.pth".format('densenet10'))
    densenet.cuda(0)
    # densenet.reset_parameters()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    optimizer = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)
    criterion = LabelSmoothingLoss(smoothing=0.1)

    for epoch in range(5):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels.type(torch.FloatTensor)
            labels = Variable(labels.cuda(0), requires_grad=True)
            inputs = Variable(inputs.cuda(0), requires_grad=True)

            optimizer.zero_grad()

            outputs = densenet(inputs)
            print('aaaaaaaaaaaaa')
            print(outputs)
            break
            # train the model with label smoothing loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    store_model(densenet, '../models/improved_densenet')


def testImprovement(criterion, testloader10, testloader, nnName, noiseMagnitude1, temper):

    densenet = load_model("../models/{}".format(nnName))
    densenet.cuda(0)

    t0 = time.time()

    h1 = open("softmax_scores/confidence_Our_In.txt", 'w')
    h2 = open("softmax_scores/confidence_Our_Out.txt", 'w')

    N = 10000
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images.cuda(0), requires_grad=True)
        outputs = densenet(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
        outputs = densenet(Variable(tempInputs))
        outputs = outputs / temper

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        h1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            t0 = time.time()

        if j == N - 1: break

    t0 = time.time()
    print("Processing out-of-distribution images")
    ###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images.cuda(0), requires_grad=True)
        outputs = densenet(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
        outputs = densenet(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        h2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            t0 = time.time()

        if j == N - 1: break


if __name__ == '__main__':
    train()
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    # ])
    #
    # testsetout = torchvision.datasets.ImageFolder("../data/{}".format('Imagenet'), transform=transform)
    # testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
    #                                             shuffle=False, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    # testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                            shuffle=False, num_workers=2)
    #
    # testImprovement(LabelSmoothingLoss(smoothing=0.3), testloaderIn, testloaderOut, 'improved_densenet', 0.0014, 1000)
    # # m.metric('improved_densenet', 'Imagenet')
