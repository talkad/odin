from copy import deepcopy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
import secrets
import time
from torch.utils.data import TensorDataset
import warnings
import csv
from model_evaluator import eval_improvement, eval_models
from metrics import error_detection, fpr, auc, aupr
from time import time


# store the model after training
def store_model(model, filename):
    pickle_file = open(filename, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()

# load the stored model
def load_model(filename):
    pickle_file = open(filename, 'rb')
    loaded = pickle.load(pickle_file)
    pickle_file.close()

    return loaded


# the suggested improvement
# replacing the cross entropy loss with label smoothing loss
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


# simplified model for training (compared the the densenet in the article)
class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# train certain model with the given criterion
def train(model, trainLoader, criterion, modelName, subset_num):

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # number of epochs

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            labels = labels - torch.ones([4], dtype=torch.int64) * 10 * subset_num

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

    # store_model(model, f'../models/{modelName}')


def evaluate_models(model_original, model_improved, hyper_params, inDistLoader, fold_num):
    results_base = {}
    results_article = {}
    result_improve = {}

    for i in range(50):
        temperature = secrets.choice(hyper_params['temperature'])
        magnitude = secrets.choice(hyper_params['magnitude'])

        if f'{temperature}_{magnitude}' not in results_base:

            print(f'evaluate temp {temperature} mag {magnitude}')
            eval_models(model_original, temperature, magnitude, inDistLoader, fold_num)
            eval_improvement(model_improved, temperature, magnitude, inDistLoader, fold_num)

            errorBase, errorNew, errorNew_Improved = error_detection(fold_num)

            results_base[f'{temperature}_{magnitude}'] = (1 - errorBase) * 100
            results_article[f'{temperature}_{magnitude}'] = (1 - errorNew) * 100
            result_improve[f'{temperature}_{magnitude}'] = (1 - errorNew_Improved) * 100

    return ((max(results_base, key=results_base.get), max(results_base.values())),
            (max(results_article, key=results_article.get), max(results_article.values())),
            (max(result_improve, key=result_improve.get), max(result_improve.values())))


# define a cross validation function
def cross_validation(model, hyper_params, criterion_original, criterion_improved, dataset, k_fold=10, csv_name='CIFAR10', subset_num=0):
    # define all table cols
    header = ['Dataset Name', 'Cross Validation [1-10]'] + ['Algorithm Name', 'best temperature_magnitude', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training TIme', 'Inference TIme', ''] * 3
    f = open(f'{csv_name}.csv', 'w', encoding='UTF8')
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

        print('train with original loss')
        original_train = time()
        train(new_model_original, train_loader, criterion_original, f'original_model_{fold_num}', subset_num)
        original_train = time() - original_train
        print('train with improved loss')
        improved_train = time()
        train(new_model_improved, train_loader, criterion_improved, f'improved_model_{fold_num}', subset_num)
        improved_train = time() - improved_train

        # new_model_original = load_model(f'../models/original_model_{fold_num}')
        # new_model_improved = load_model(f'../models/improved_model_{fold_num}')

        (baseAcc), (articleAcc), (improveAcc) = evaluate_models(new_model_original, new_model_improved, hyper_params, test_loader, fold_num)
        fprBase, fprNew, fprImproved = fpr(fold_num)
        aurocBase, aurocNew, aurocImproved = auc(fold_num)
        auprBase, auprNew, auprImproved = aupr(fold_num)

        data = [csv_name, fold_num] +\
               ['base', baseAcc[0], baseAcc[1], 95, fprBase, 95 / (95 + fprBase), aurocBase, auprBase, original_train, '', ''] +\
               ['article', articleAcc[0], articleAcc[1], 95, fprNew, 95 / (95 + fprBase), aurocNew, auprNew, original_train, '', ''] +\
               ['improve', improveAcc[0], improveAcc[1], 95, fprImproved, 95 / (95 + fprBase), aurocImproved, auprImproved, improved_train, '', '']
        writer.writerow(data)
        fold_num += 1

    f.close()
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # update according to the article
    space = dict()
    space['temperature'] = [10, 20, 50, 100, 200, 500, 1000, 1500]
    space['magnitude'] = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]

    # transform the in-out images to be of the same shape
    transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 90)),
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    for i in range(9):
        trainset = torch.utils.data.ConcatDataset([trainset, torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)])

    indexes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for idx, (_, feature) in enumerate(trainset):
        indexes[int(feature / 10)].append(idx)

    split_cifar = []

    for i in range(10):
        split_cifar.append(torch.utils.data.dataset.Subset(trainset, indexes[i]))

    criterion_original = nn.CrossEntropyLoss()
    criterion_improved = LabelSmoothingLoss(smoothing=0.1)
    convnet = Convnet()

    for i, subset in enumerate(split_cifar):
        cross_validation(convnet, space, criterion_original, criterion_improved, subset, csv_name=f'CIFAR{i}', subset_num=i)


