from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


def nested_cross_validation(model, hyper_params, X_features, y_labels):
    external_k = 10
    internal_k = 3

    best_f1 = 0
    best_hyperParams = {}

    cv_outer = KFold(n_splits=external_k, shuffle=True)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X_features):

        X_train, X_test = X_features[train_ix, :], X_features[test_ix, :]
        y_train, y_test = y_labels[train_ix], y_labels[test_ix]

        cv_inner = KFold(n_splits=internal_k, shuffle=True)

        new_model = deepcopy(model)

        search = RandomizedSearchCV(new_model, hyper_params, scoring='f1_macro', cv=cv_inner, refit=True, n_iter=50)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        yhat = best_model.predict(X_test)
        f1 = f1_score(y_test, yhat, average='macro')

        if f1 > best_f1:
            best_f1 = f1
            best_hyperParams = result.best_params_

        outer_results.append(accuracy_score(y_test, yhat))
        print(
            f'accuracy: {accuracy_score(y_test, yhat)}  |  f1: {f1_score(y_test, yhat, average="macro")}  |  AUROC: {roc_auc_score(y_test, yhat)}')

    return best_hyperParams


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == '__main__':

    # build densenet trained on CIFAR10 and test against ImageNet
    space = dict()
    space['temperature'] = [x for x in range(800, 1250, 50)]
    space['max_depth'] = [x/10000 for x in range(10, 20, 1)]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

    testsetout = torchvision.datasets.ImageFolder("../data/{}".format('Imagenet'), transform=transform)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                                shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                               shuffle=False, num_workers=2)

    densenet = torch.load("../models/{}.pth".format('densenet10'))
    densenet.cuda(0)
    # reset model params
    densenet.apply(weight_reset)

