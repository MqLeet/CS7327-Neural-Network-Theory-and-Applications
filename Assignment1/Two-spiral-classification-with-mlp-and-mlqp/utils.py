#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/10 16:43     mql        1.0         None
'''
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)

def sigmoid(x, derive = False):
    if derive == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(- x))



def dataloader(mode):
    file_path = "data/two_spiral_{}_data.txt".format(mode)
    res = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            temp = line.split()
            res.append([np.array([float(temp[0]), float(temp[1])]), int(temp[2])])

    return np.array(res)

def plot_decision_boundary(model, x, y, file_dir):
    # preprocess: x_min: -6.0, x_max: 6.0, y_min: -5.50887, y_max: 5.50887
    x_min, x_max = -7, 7
    y_min, y_max = -7, 7
    step = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    plot_data = np.c_[xx.ravel(), yy.ravel()]
    plot_data_processed = []
    for item in plot_data:
        tmp = model(item)
        plot_data_processed.append(tmp)

    plot_data_processed = np.array(plot_data_processed)
    plot_data_processed = plot_data_processed.reshape(xx.shape)

    plt.figure()
    plt.title('decision boundary on test set')
    plt.contourf(xx, yy, plot_data_processed, cmap=plt.cm.Spectral)
    plt.ylabel('y')
    plt.xlabel('x')
    scatter = plt.scatter(x[0, :], x[1, :], c = y, cmap=plt.cm.Spectral)
    plt.legend(handles = scatter.legend_elements()[0], labels=['label=0', 'label=1'], loc='upper right')
    plt.colorbar()
    plt.savefig(file_dir)

def plot_acc_curve(df, file_dir):
    plt.figure()
    plt.title("accuracy curve")

    train_acc = df['train_acc']
    test_acc = df['test_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, color= 'blue', label='train accuracy')
    plt.plot(epochs, test_acc, color='orange', label='test accuracy')
    plt.legend()
    plt.savefig(file_dir)



def plot_loss_curve(df, file_dir):
    plt.figure()
    plt.title("loss curve")

    train_loss = df['train_loss']
    test_loss = df['test_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, color= 'blue', label='train loss')
    plt.plot(epochs, test_loss, color='orange', label='test loss')
    plt.legend()
    plt.savefig(file_dir)



if __name__ == "__main__":
    train_dataset = dataloader(mode="test")
    x_min = float('inf')
    x_max = -float('inf')
    y_min = float('inf')
    y_max = -float('inf')
    for item in train_dataset:
        x_min = min(item[0][0], x_min)
        x_max = max(item[0][0], x_max)

        y_min = min(item[0][1], y_min)
        y_max = max(item[0][1], y_max)

    print("x_min:", x_min)
    print("x_max:", x_max)
    print("y_min:", y_min)
    print("y_max:", y_max)