from copy import deepcopy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import torch.optim as optim
import secrets
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import warnings
import operator
import csv


def store_model(model, filename):
    pickle_file = open(filename, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()


def load_model(filename):
    pickle_file = open(filename, 'rb')
    loaded = pickle.load(pickle_file)
    pickle_file.close()

    return loaded


# the suggested improvement
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


class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def eval_improvement(model_improved, temper, noiseMagnitude1, testloader, fold_num):

    t0 = time.time()
    criterion_improved = LabelSmoothingLoss(smoothing=0.1)
    h1 = open(f"softmax_scores/confidence_Our_In_Improved{fold_num}.txt", 'w')
    h2 = open(f"softmax_scores/confidence_Our_Out_Improved{fold_num}.txt", 'w')

    N = 10000
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images, requires_grad=True)
        outputs = model_improved(inputs)

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
        labels = Variable(torch.LongTensor([maxIndexTemp]))
        loss = criterion_improved(outputs, labels)
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
        outputs = model_improved(Variable(tempInputs))
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

        inputs = Variable(images, requires_grad=True)
        outputs = model_improved(inputs)

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
        labels = Variable(torch.LongTensor([maxIndexTemp]))
        loss = criterion_improved(outputs, labels)
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
        outputs = model_improved(Variable(tempInputs))
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


def eval_models(model_original, temper, noiseMagnitude1, testloader, fold_num):

    t0 = time.time()
    f1 = open(f"./softmax_scores/confidence_Base_In{fold_num}.txt", 'w')
    f2 = open(f"./softmax_scores/confidence_Base_Out{fold_num}.txt", 'w')
    g1 = open(f"./softmax_scores/confidence_Our_In{fold_num}.txt", 'w')
    g2 = open(f"./softmax_scores/confidence_Our_Out{fold_num}.txt", 'w')

    testsetout = torchvision.datasets.ImageFolder("../data/{}".format('Imagenet'), transform=transform)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                                shuffle=False, num_workers=2)

    criterion_original = nn.CrossEntropyLoss()

    N = 10000
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images, requires_grad=True)
        outputs = model_original(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]))
        loss = criterion_original(outputs, labels)
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
        outputs = model_original(Variable(tempInputs))
        outputs = outputs / temper

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            t0 = time.time()

        if j == N - 1: break

    t0 = time.time()
    print("Processing out-of-distribution images")
    ###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloaderOut):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images, requires_grad=True)
        outputs = model_original(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]))
        loss = criterion_original(outputs, labels)
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
        outputs = model_original(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1 - 1000, N - 1000, time.time() - t0))
            t0 = time.time()

        if j == N - 1: break


def train(model, trainLoader, criterion, modelName):

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # number of epochs

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
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

    store_model(model, f'../models/{modelName}')


def error_detection(name, fold_num):
    # calculate the minimum detection error
    # calculate baseline
    T = 1
    cifar = np.loadtxt(f'./softmax_scores/confidence_Base_In{fold_num}.txt', delimiter=',')
    other = np.loadtxt(f'./softmax_scores/confidence_Base_Out{fold_num}.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(f'./softmax_scores/confidence_Our_In{fold_num}.txt', delimiter=',')
    other = np.loadtxt(f'./softmax_scores/confidence_Our_Out{fold_num}.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    # calculate our improved algorithm
    T = 1000
    cifar = np.loadtxt(f'./softmax_scores/confidence_Our_In_Improved{fold_num}.txt', delimiter=',')
    other = np.loadtxt(f'./softmax_scores/confidence_Our_Out_Improved{fold_num}.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorNew_Improved = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew_Improved = np.minimum(errorNew_Improved, (tpr + error2) / 2.0)

    return errorBase, errorNew, errorNew_Improved


def evaluate_models(model_original, model_improved, hyper_params, inDistLoader, fold_num):
    # eval_models(model_original, 1000, 0.0014, inDistLoader, fold_num)
    # eval_improvement(model_improved, 1000, 0.0014, inDistLoader, fold_num)
    #
    # errorBase, errorNew, errorNew_Improved = error_detection("CIFAR-10", fold_num)
    # print(f"Accuracy: {(1 - errorBase) * 100}, {(1 - errorNew) * 100}, {(1 - errorNew_Improved) * 100}")
    #
    # return (('as', (1 - errorBase) * 100),
    #         ('sd', (1 - errorNew) * 100),
    #         ('df', (1 - errorNew_Improved) * 100))

    results_base = {}
    results_article = {}
    result_improve = {}

    for i in range(5):  ## change it to 50!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        temperature = secrets.choice(hyper_params['temperature'])
        magnitude = secrets.choice(hyper_params['magnitude'])

        if f'{temperature}_{magnitude}' not in results_base:

            print(f'evaluate temp {temperature} mag {magnitude}')
            eval_models(model_original, temperature, magnitude, inDistLoader, fold_num)
            eval_improvement(model_improved, temperature, magnitude, inDistLoader, fold_num)

            errorBase, errorNew, errorNew_Improved = error_detection("CIFAR-10", fold_num)

            results_base[f'{temperature}_{magnitude}'] = (1 - errorBase) * 100
            results_article[f'{temperature}_{magnitude}'] = (1 - errorNew) * 100
            result_improve[f'{temperature}_{magnitude}'] = (1 - errorNew_Improved) * 100

    return ((max(results_base, key=results_base.get), max(results_base.values())),
            (max(results_article, key=results_article.get), max(results_article.values())),
            (max(result_improve, key=result_improve.get), max(result_improve.values())))


# define a cross validation function
def cross_validation(model, hyper_params, criterion_original, criterion_improved, dataset, k_fold=10):

    header = ['Dataset Name'] + ['Algorithm Name', 'Cross Validation [1-10]', 'best temperature_magnitude', 'Acc', ''] * 3
    f = open('CIFAR10.csv', 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(header)

    fold_num = 1

    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)

    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        testl = trlr
        testr = i * seg + seg
        trrl = testr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))

        train_indices = train_left_indices + train_right_indices
        test_indices = list(range(testl, testr))

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        test_set = torch.utils.data.dataset.Subset(dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                                   shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

        new_model_original = deepcopy(model)
        new_model_improved = deepcopy(model)

        # print('train with original loss')
        # train(new_model_original, train_loader, criterion_original, f'original_model_{fold_num}')
        # print('train with improved loss')
        # train(new_model_improved, train_loader, criterion_improved, f'improved_model_{fold_num}')

        new_model_original = load_model(f'../models/original_model_{fold_num}')
        new_model_improved = load_model(f'../models/improved_model_{fold_num}')

        (base), (article), (improve) = evaluate_models(new_model_original, new_model_improved, hyper_params, test_loader, fold_num)

        data = ['CIFAR10'] + ['base', fold_num, base[0], f'{base[1]}%', ''] + ['article', fold_num, article[0], f'{article[1]}%', ''] + ['improve', fold_num, improve[0], f'{improve[1]}%', '']
        writer.writerow(data)
        break
        fold_num += 1

    f.close()
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # build convnet trained on CIFAR10 and test against ImageNet
    space = dict()
    space['temperature'] = [x for x in range(800, 1250, 50)]
    space['magnitude'] = [x/10000 for x in range(10, 20, 1)]

    # transform the in-out images to be of the same shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(32),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    criterion_original = nn.CrossEntropyLoss()
    criterion_improved = LabelSmoothingLoss(smoothing=0.1)
    convnet = Convnet()

    cross_validation(convnet, space, criterion_original, criterion_improved, trainset)

