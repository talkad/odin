import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
from torch.utils.data import TensorDataset
from evaluator import LabelSmoothingLoss


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