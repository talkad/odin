from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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


class Convnet_CIFAR10(nn.Module):
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


def testImprovement(model_improved, temper, noiseMagnitude1, testloader, fold_num):

    t0 = time.time()
    criterion_improved = LabelSmoothingLoss(smoothing=0.1)
    h1 = open(f"softmax_scores/confidence_Our_In{fold_num}.txt", 'w')
    h2 = open(f"softmax_scores/confidence_Our_Out{fold_num}.txt", 'w')

    N = 10000
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader):
        if j < 1000: continue
        images, _ = data

        inputs = Variable(images.cuda(0), requires_grad=True)
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
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
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

        inputs = Variable(images.cuda(0), requires_grad=True)
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
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
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

        inputs = Variable(images.cuda(0), requires_grad=True)
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
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
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

        inputs = Variable(images.cuda(0), requires_grad=True)
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
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(0))
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


def evaluate_models(model_original, model_improved, hyper_params, inDistLoader, fold_num):
    results = {}
    for i in range(50):
        temperature = secrets.choice(hyper_params['temperature'])
        magnitude = secrets.choice(hyper_params['magnitude'])

        if f'temp{temperature}mag{magnitude}' not in results:
            print(f'evaluate temp {temperature} mag {magnitude}')
            acc_base, acc_original, acc_improved = eval_models(model_original, model_improved, temperature, magnitude, inDistLoader, fold_num)
            results[f'temp{temperature}mag{magnitude}'] = (acc_base, acc_original, acc_improved)

    return 0


# define a cross validation function
def cross_validation(model, hyper_params, criterion_original, criterion_improved, dataset, k_fold=10):
    fold_num = 1

    train_score = pd.Series()
    val_score = pd.Series()

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

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                                   shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

        new_model_original = deepcopy(model)
        new_model_improved = deepcopy(model)

        train(new_model_original, train_loader, criterion_original, f'original_model_{fold_num}')
        train(new_model_improved, train_loader, criterion_improved, f'improved_model_{fold_num}')

        best_hyperParams = evaluate_models(new_model_original, new_model_improved, hyper_params, test_loader, fold_num)
        print('aaaaaaaaaaaaaaaa')
        print(best_hyperParams)
        break

        fold_num += 1

    return train_score, val_score


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # build densenet trained on CIFAR10 and test against ImageNet
    space = dict()
    space['temperature'] = [x for x in range(800, 1250, 50)]
    space['magnitude'] = [x/10000 for x in range(10, 20, 1)]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    criterion_original = nn.CrossEntropyLoss()
    criterion_improved = LabelSmoothingLoss(smoothing=0.1)
    convnet = Convnet_CIFAR10()

    cross_validation(convnet, space, criterion_original, criterion_improved, trainset)

    # X_features, y_labels = [np.array(X) for (X, _) in loader], [np.array(y) for (_, y) in loader]
    #
    # # # cross_validation(convnet, space, criterion_original, criterion_improved, X_features, y_labels)
    #
    # trainset2 = TensorDataset(torch.tensor(X_features), torch.tensor(y_labels))
    #
    # trainloader = torch.utils.data.DataLoader(trainset2, batch_size=batch_size, shuffle=True, num_workers=2)
    # for data in trainloader:
    #     print(data)
    #     break
    #
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # for data in trainloader:
    #     print(data)
    #     break
    # train(convnet, trainloader, criterion_original, '')




















# def cross_validation2(model, hyper_params, criterion_original, criterion_improved, X_features, y_labels):
#     external_k = 10
#
#     cv_outer = KFold(n_splits=external_k, shuffle=True)
#     fold_num = 1
#
#     # enumerate splits
#     for train_ix, test_ix in cv_outer.split(X_features):
#
#         X_train, X_test = X_features[train_ix], X_features[test_ix]
#         y_train, y_test = y_labels[train_ix], y_labels[test_ix]
#
#         new_model_original = deepcopy(model)
#         new_model_improved = deepcopy(model)
#
#         trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32,
#                                                   shuffle=True, num_workers=2)
#
#         print('training models')
#         train(new_model_original, trainloader, criterion_original, f'original_model_{fold_num}')
#         train(new_model_improved, trainloader, criterion_improved, f'improved_model_{fold_num}')
#
#         break
#
#         fold_num += 1
#
#         # evaluate the models
#         testloader = torch.utils.data.DataLoader(zip(X_test, y_test), batch_size=batch_size,
#                                                   shuffle=True, num_workers=2)
#
#         best_hyperParams = evaluate_models(new_model_original, new_model_improved, hyper_params, testloader)
#         print(best_hyperParams)
#         break
#
#     return best_hyperParams
