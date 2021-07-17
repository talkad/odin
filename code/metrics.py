import numpy as np

# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

# the code in this class changed to be suitable to our use case


# calculate the detection error
def error_detection(fold_num):
    # calculate baseline
    base_in = np.loadtxt(f'./softmax_scores/confidence_Base_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Base_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 1

    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # calculate our algorithm
    start = 0.1
    end = 0.12

    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out{fold_num}.txt', delimiter=',')

    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    # calculate our improved algorithm
    start = 0.1
    end = 0.12

    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In_Improved{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out_Improved{fold_num}.txt', delimiter=',')

    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    errorNew_Improved = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew_Improved = np.minimum(errorNew_Improved, (tpr + error2) / 2.0)

    return errorBase, errorNew, errorNew_Improved


# calculate the falsepositive error when tpr is 95%
def fpr(fold_num):
    # calculate baseline
    base_in = np.loadtxt(f'./softmax_scores/confidence_Base_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Base_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 1
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 1 and tpr >= 0.9:
            fpr += error2
            total += 1
    fprBase = fpr / total

    # calculate our algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 1 and tpr >= 0.9:
            fpr += error2
            total += 1
    fprNew = fpr / total

    # calculate improved algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In_Improved{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out_Improved{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 1 and tpr >= 0.9:
            fpr += error2
            total += 1
    fprImproved = fpr / total

    return fprBase*100, fprNew*100, fprImproved*100


# calculate the AUC
def auc(fold_num):
    # calculate baseline
    base_in = np.loadtxt(f'./softmax_scores/confidence_Base_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Base_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 1
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    # calculate our algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    aurocNew = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocNew += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocNew += fpr * tpr

    # calculate improved algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In_Improved{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out_Improved{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12
    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    aurocImproved = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocImproved += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocImproved += fpr * tpr

    return aurocBase*100, aurocNew*100, aurocImproved*100


# calculate the AUPR
def aupr(fold_num):
    # calculate baseline
    base_in = np.loadtxt(f'./softmax_scores/confidence_Base_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Base_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 1

    gap = (end - start) / 100000
    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # calculate our algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12

    gap = (end - start) / 100000

    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision

    # calculate improved algorithm
    base_in = np.loadtxt(f'./softmax_scores/confidence_Our_In_Improved{fold_num}.txt', delimiter=',')
    base_out = np.loadtxt(f'./softmax_scores/confidence_Our_Out_Improved{fold_num}.txt', delimiter=',')

    start = 0.1
    end = 0.12

    gap = (end - start) / 100000

    Y1 = base_out[:, 2]
    X1 = base_in[:, 2]
    auprImproved = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprImproved += (recallTemp - recall) * precision
        recallTemp = recall
    auprImproved += recall * precision

    return auprBase*100, auprNew*100, auprImproved*100
