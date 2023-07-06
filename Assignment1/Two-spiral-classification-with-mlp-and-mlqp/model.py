#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/10 18:39     mql        1.0         None
'''
import numpy as np
from utils import sigmoid, dataloader, set_seed
import tqdm

class MLQP:
    def __init__(self, num_hidden_layers, input_dim, output_dim, hidden_dim):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.parameters = dict()
        self.gradients = dict()

        self.layer_tmp = dict()

        # initialize network layers
        # k == 1 means input layer, k == num_hidden_layers + 1 means output_layer
        for k in range(1, num_hidden_layers + 2):
            self.parameters[k] = dict()
            if k == 1:
                self.parameters[k]['u'] = np.random.normal(size=(self.input_dim, self.hidden_dim))
                self.parameters[k]['v'] = np.random.normal(size=(self.input_dim, self.hidden_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.hidden_dim, 1))

            elif k == num_hidden_layers + 1:
                self.parameters[k]['u'] = np.random.normal(size=(self.hidden_dim, self.output_dim))
                self.parameters[k]['v'] = np.random.normal(size=(self.hidden_dim, self.output_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.output_dim, 1))

            else:
                self.parameters[k]['u'] = np.random.normal(size=(self.hidden_dim, self.hidden_dim))
                self.parameters[k]['v'] = np.random.normal(size=(self.hidden_dim, self.hidden_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.hidden_dim, 1))

        # store the temporary result
        for k in range(0, self.num_hidden_layers + 2):
            self.layer_tmp[k] = dict()


    # input has size [2]
    def forward(self, input):
        self.layer_tmp[0]['x'] = input[:, np.newaxis]
        for k in range(1, self.num_hidden_layers + 2):
            self.layer_tmp[k]['n'] = np.dot(self.parameters[k]['u'].T, self.layer_tmp[k - 1]['x'] ** 2) \
                                    + np.dot(self.parameters[k]['v'].T, self.layer_tmp[k - 1]['x']) \
                                    + self.parameters[k]['b']

            self.layer_tmp[k]['x'] = sigmoid(self.layer_tmp[k]['n'], derive=False)

        return self.layer_tmp[self.num_hidden_layers + 1]['x']

    def backward(self, y):
        # MSE loss
        self.gradients[self.num_hidden_layers + 1] = y - self.layer_tmp[self.num_hidden_layers + 1]['x']
        for k in range(self.num_hidden_layers, 0, -1):
            delta_net = sigmoid(self.layer_tmp[k + 1]['x'], derive=True)
            self.gradients[k] = 2 * self.layer_tmp[k]['x'] * np.dot(self.parameters[k + 1]['u'], delta_net * self.gradients[k + 1]) + \
                                np.dot(self.parameters[k + 1]['v'], delta_net * self.gradients[k + 1])


    def step(self, lr = 1e-3):
        for k in range(1, self.num_hidden_layers + 2):
            delta_n = sigmoid(self.layer_tmp[k]['x'], derive=True)
            self.parameters[k]['u'] += lr * np.outer(self.layer_tmp[k-1]['x'] ** 2, delta_n * self.gradients[k])
            self.parameters[k]['v'] += lr * np.outer(self.layer_tmp[k-1]['x'], delta_n * self.gradients[k])
            self.parameters[k]['b'] += delta_n * lr * self.gradients[k]

class MLP:
    def __init__(self, num_hidden_layers, input_dim, output_dim, hidden_dim):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.parameters = dict()
        self.gradients = dict()

        self.layer_tmp = dict()

        # initialize network layers
        # k == 1 means input layer, k == num_hidden_layers + 1 means output_layer
        for k in range(1, num_hidden_layers + 2):
            self.parameters[k] = dict()
            if k == 1:
                self.parameters[k]['w'] = np.random.normal(size=(self.input_dim, self.hidden_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.hidden_dim, 1))

            elif k == num_hidden_layers + 1:
                self.parameters[k]['w'] = np.random.normal(size=(self.hidden_dim, self.output_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.output_dim, 1))

            else:
                self.parameters[k]['w'] = np.random.normal(size=(self.hidden_dim, self.hidden_dim))
                self.parameters[k]['b'] = np.random.normal(size=(self.hidden_dim, 1))

        # store the temporary result
        for k in range(0, self.num_hidden_layers + 2):
            self.layer_tmp[k] = dict()

    # input has size [2]
    def forward(self, input):
        self.layer_tmp[0]['x'] = input[:, np.newaxis]
        for k in range(1, self.num_hidden_layers + 2):
            self.layer_tmp[k]['n'] = np.dot(self.parameters[k]['w'].T, self.layer_tmp[k - 1]['x']) \
                                    + self.parameters[k]['b']

            self.layer_tmp[k]['x'] = sigmoid(self.layer_tmp[k]['n'], derive=False)

        return self.layer_tmp[self.num_hidden_layers + 1]['x']

    def backward(self, y):
        # MSE loss
        self.gradients[self.num_hidden_layers + 1] = y - self.layer_tmp[self.num_hidden_layers + 1]['x']
        for k in range(self.num_hidden_layers, 0, -1):
            delta_net = sigmoid(self.layer_tmp[k + 1]['x'], derive=True)
            self.gradients[k] = np.dot(self.parameters[k + 1]['w'], delta_net * self.gradients[k + 1])

    def step(self, lr = 1e-3):
        for k in range(1, self.num_hidden_layers + 2):
            delta_n = sigmoid(self.layer_tmp[k]['x'], derive=True)
            self.parameters[k]['w'] += lr * np.outer(self.layer_tmp[k-1]['x'], delta_n * self.gradients[k])
            self.parameters[k]['b'] += delta_n * lr * self.gradients[k]


if __name__ == "__main__":
    train_data = dataloader(mode="train")
    test_data = dataloader(mode="test")

    set_seed(seed=42)
    mlqp = MLQP(num_hidden_layers=1, input_dim=2, output_dim=1, hidden_dim=32)
    mlp = MLP(num_hidden_layers=1, input_dim=2, output_dim=1, hidden_dim=32)

    for epoch in tqdm.tqdm(range(2000)):
        preds = []
        for coordinates, label in train_data:
            pred = mlp.forward(coordinates)
            mlp.backward(label)
            mlp.step(lr=1e-1)
            preds.append(pred)

        preds = np.squeeze(np.array(preds))
        loss = (0.5 * (train_data[:, 1] - preds) ** 2).sum()
        acc = ((preds > 0.5) == train_data[:, 1]).mean()
        # print("*************************loss in epoch_{} is: {}".format(epoch, loss))
        print("*************************acc in epoch_{} is: {}".format(epoch, acc))





