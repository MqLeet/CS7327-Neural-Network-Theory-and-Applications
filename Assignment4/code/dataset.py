#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/28 13:48     mql        1.0         None
'''
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SEED_IV_DATASET(Dataset):
    def __init__(self, people, is_train, use_geometry=False):
        super(SEED_IV_DATASET, self).__init__()
        self.data = []
        self.label = []

        for session in range(1, 4):
            names = os.listdir("./data/" + str(session))
            for name in names:
                if is_train:
                    if eval(name.split("_")[0]) != people:
                        real_dir = os.path.join("./data", str(session), name)
                        self.data.append(np.load(os.path.join(real_dir, "train_data.npy")))
                        self.label.append(np.load(os.path.join(real_dir, "train_label.npy")))
                        self.data.append(np.load(os.path.join(real_dir, "test_data.npy")))
                        self.label.append(np.load(os.path.join(real_dir, "test_label.npy")))
                if not is_train:
                    if eval(name.split("_")[0]) == people:
                        real_dir = os.path.join("./data", str(session), name)
                        self.data.append(np.load(os.path.join(real_dir, "train_data.npy")))
                        self.label.append(np.load(os.path.join(real_dir, "train_label.npy")))
                        self.data.append(np.load(os.path.join(real_dir, "test_data.npy")))
                        self.label.append(np.load(os.path.join(real_dir, "test_label.npy")))

        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        if use_geometry:
            self.data = self.info_transform()

    def __getitem__(self, index):
        return np.float32(self.data[index]), np.int64(self.label[index])

    def __len__(self):
        return self.data.shape[0]

    def info_transform(self):
        data_size = self.data.shape[0]
        x_train_new = torch.zeros(data_size, 81, 5)
        x_train = torch.from_numpy(self.data)
        pos_list = [2, 4, 6, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                    64, 65, 67, 69, 70, 71, 73, 74, 76, 78, 79]
        x_train_new[:, pos_list, :] = x_train.float()
        transformed_data = (x_train_new.reshape(-1, 9, 9, 5)).permute(0, 3, 1, 2) # size * 5 * 9 * 9
        return transformed_data


if __name__ == "__main__":
    people_list = list(range(1, 16))
    data_root_dir = os.path.join(os.getcwd(), "SEED-IV")

    for people in people_list:
        train_dataset = SEED_IV_DATASET(people=people, is_train=True, use_geometry=True)
        test_dataset = SEED_IV_DATASET(people=people, is_train=False, use_geometry=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)


        for index, item in enumerate(train_dataloader):
            data, label = item
            print(data.shape)
            print(label.shape)
            print(X)


