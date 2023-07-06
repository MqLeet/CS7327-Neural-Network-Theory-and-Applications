#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/11 16:46     mql        1.0         None
'''

import os
import time
from utils import *
from model import MLP, MLQP
import tqdm
import pandas as pd

def train_mlp(num_hidden_layers, input_dim, output_dim, hidden_dim,
              name, epoches = 1000, lr=1e-1):
    train_data = dataloader(mode="train")
    test_data = dataloader(mode="test")

    mlp = MLP(num_hidden_layers=num_hidden_layers, input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    log_dir = os.path.join("mlp_logs", name)
    os.makedirs(os.path.join(log_dir, "visualize"), exist_ok=True)

    time_start = time.time()
    r_time = 0
    train_loss, train_acc, test_loss, test_acc, li_time = [], [], [], [], []

    with tqdm.tqdm(range(1, epoches), desc="Training Process") as pbar:
        for epoch in pbar:
            preds = []
            for coordinates, label in train_data:
                pred = mlp.forward(coordinates)
                mlp.backward(label)
                mlp.step(lr)
                preds.append(pred)

            preds = np.squeeze(np.array(preds))
            loss = (0.5 * (train_data[:, 1] - preds) ** 2).sum()
            acc = ((preds > 0.5) == train_data[:, 1]).mean()
            pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)

            # test
            if epoch % 10 == 0:
                # training log
                train_loss.append(loss)
                train_acc.append(acc)

                # testing log
                preds = []
                for coordinates, label in test_data:
                    pred = mlp.forward(coordinates)
                    preds.append(pred)
                preds = np.squeeze(np.array(preds))
                loss_test = (0.5 * (test_data[:, 1] - preds) ** 2).sum()
                acc_test = ((preds > 0.5) == test_data[:, 1]).mean()
                pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
                test_loss.append(loss_test)
                test_acc.append(acc_test)

                # time log
                li_time.append(time.time() - time_start - r_time)

            if epoch % 1000 == 0:
                r_time0 = time.time()
                # plot decision boundary
                plot_decision_boundary(lambda x: mlp.forward(x), np.stack(train_data[:, 0]).T, train_data[:, 1],
                                       os.path.join(log_dir, "visualize", "epoch_{}".format(epoch)))
                r_time += time.time() - r_time0

    dataframe = pd.DataFrame(
        {'time': li_time, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
         'test_acc': test_acc})
    dataframe.to_csv(os.path.join(log_dir, "log.csv"), sep=',')
    plot_acc_curve(dataframe,
                   os.path.join(log_dir, "visualize", "curve_acc"))

    plot_loss_curve(dataframe,
                    os.path.join(log_dir, "visualize", "loss_acc"))


def train_mlqp(num_hidden_layers, input_dim, output_dim, hidden_dim,
              name, epoches = 1000, lr=1e-1):
    train_data = dataloader(mode="train")
    test_data = dataloader(mode="test")

    mlqp = MLQP(num_hidden_layers=num_hidden_layers, input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    log_dir = os.path.join("mlqp_logs", name)
    os.makedirs(os.path.join(log_dir, "visualize"), exist_ok=True)

    time_start = time.time()
    r_time = 0
    train_loss, train_acc, test_loss, test_acc, li_time = [], [], [], [], []

    with tqdm.tqdm(range(1, epoches), desc="Training Process") as pbar:
        for epoch in pbar:
            preds = []
            for coordinates, label in train_data:
                pred = mlqp.forward(coordinates)
                mlqp.backward(label)
                mlqp.step(lr)
                preds.append(pred)

            preds = np.squeeze(np.array(preds))
            loss = (0.5 * (train_data[:, 1] - preds) ** 2).sum()
            acc = ((preds > 0.5) == train_data[:, 1]).mean()
            pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)

            # test
            if epoch % 10 == 0:
                # training log
                train_loss.append(loss)
                train_acc.append(acc)

                # testing log
                preds = []
                for coordinates, label in test_data:
                    pred = mlqp.forward(coordinates)
                    preds.append(pred)
                preds = np.squeeze(np.array(preds))
                loss_test = (0.5 * (test_data[:, 1] - preds) ** 2).sum()
                acc_test = ((preds > 0.5) == test_data[:, 1]).mean()
                pbar.set_postfix(loss='%.4f' % loss, acc='%.4f' % acc)
                test_loss.append(loss_test)
                test_acc.append(acc_test)

                # time log
                li_time.append(time.time() - time_start - r_time)

            if epoch % 1000 == 0:
                r_time0 = time.time()
                # plot decision boundary
                plot_decision_boundary(lambda x: mlqp.forward(x), np.stack(test_data[:, 0]).T, test_data[:, 1],
                                       os.path.join(log_dir, "visualize", "epoch_{}".format(epoch)))
                r_time += time.time() - r_time0

    dataframe = pd.DataFrame(
        {'time': li_time, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
         'test_acc': test_acc})
    dataframe.to_csv(os.path.join(log_dir, "log.csv"), sep=',')

    plot_acc_curve(dataframe,
                   os.path.join(log_dir, "visualize", "curve_acc"))

    plot_loss_curve(dataframe,
                    os.path.join(log_dir, "visualize", "loss_acc"))

if __name__ == "__main__":
    train_mlqp(num_hidden_layers=1, input_dim=2, output_dim=1, hidden_dim=32, epoches=200, lr=1e-1, name="mlqplr1e-1")
