#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/31 14:52     mql        1.0         None
'''
import numpy as np
import random
import os
import torch
import pickle
import pandas
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

class SEED_IV_DATASET(Dataset):
    def __init__(self, people, is_train, use_geometry):
        super(SEED_IV_DATASET, self).__init__()
        self.data = []
        self.label = []
        for session in range(1,4):
            names = os.listdir("./SEED-IV/" + str(session))
            for name in names:
                if is_train:
                    if eval(name.split("_")[0]) != people:
                        real_dir = "./SEED-IV/" + str(session) + "/" + name
                        self.data.append(np.load(real_dir + "/train_data.npy"))
                        self.label.append(np.load(real_dir + "/train_label.npy"))
                        self.data.append(np.load(real_dir + "/test_data.npy"))
                        self.label.append(np.load(real_dir + "/test_label.npy"))
                if not is_train:
                    if eval(name.split("_")[0]) == people:
                        real_dir = "./SEED-IV/" + str(session) + "/" + name
                        self.data.append(np.load(real_dir + "/train_data.npy"))
                        self.label.append(np.load(real_dir + "/train_label.npy"))
                        self.data.append(np.load(real_dir + "/test_data.npy"))
                        self.label.append(np.load(real_dir + "/test_label.npy"))
        self.data = np.concatenate(self.data,axis=0)
        self.label = np.concatenate(self.label,axis=0)
        if use_geometry:
            self.data = self.info_transform()


    def __getitem__(self, index):
        #读第index行
        output = {}
        output["data"] = np.float32(self.data[index])
        output["label"] = np.int64(self.label[index])
        return output

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
        x = (x_train_new.reshape(-1, 9, 9, 5).permute(0, 3, 1, 2))
        return x

def info_transform(x_train):
    data_size = x_train.shape[0]
    x_train_new = torch.zeros(data_size, 81, 5)
    x_train = torch.from_numpy(x_train)
    pos_list = [2, 4, 6, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 67, 69, 70, 71, 73, 74, 76, 78, 79]
    x_train_new[:, pos_list, :] = x_train.float()
    x = (x_train_new.reshape(-1, 9, 9, 5).permute(0, 3, 1, 2))
    return x



def load_one_subject(data_root_dir,
         session,
         subject,
         is_train=True):
    session_list = os.listdir(os.path.join(data_root_dir, str(session)))
    for folder in session_list:
        if folder.split(sep='_')[0] == str(subject):
            train_data = np.load(os.path.join(data_root_dir, str(session),folder, "train_data.npy"))
            train_label = np.load(os.path.join(data_root_dir, str(session), folder, "train_label.npy"))
            test_data = np.load(os.path.join(data_root_dir, str(session), folder, "test_data.npy"))
            test_label = np.load(os.path.join(data_root_dir, str(session), folder, "test_label.npy"))

            if is_train:
                return train_data, train_label
            else:
                return test_data, test_label

def load_all_sessions(data_root_dir,
                      subject,
                      is_train=True):
    session_list = [1, 2, 3]
    data_all = []
    label_all = []

    for session in session_list:
        dir_list = os.listdir(os.path.join(data_root_dir, str(session)))
        for folder in dir_list:
            if folder.split("_")[0] == subject:
                real_dir = os.path.join(data_root_dir, str(session), folder)
                if is_train:
                    data = np.load(os.path.join(real_dir, "./train_data.npy"))
                    label = np.load(os.path.join(real_dir, "./train_label.npy"))
                else:
                    data = np.load(os.path.join(real_dir, "./test_data.npy"))
                    label = np.load(os.path.join(real_dir, "./test_label.npy"))
                data_all.append(data)
                label_all.append(label)
    data_all = np.concatenate(data_all,axis=0)
    label_all = np.concatenate(label_all,axis=0)
    return data_all, label_all

BLUE = (0.12156862745098039, 0.47058823529411764, 0.7058823529411765)
RED = (0.8901960784313725, 0.10196078431372549, 0.10980392156862745)

def visualize_acc(acc_list,save_dir,name,color=RED):
    acc_list = np.array(acc_list).reshape(-1)
    acc_list = np.sort(acc_list)[::-1]
    plt.figure(figsize=(6,6))
    plt.hist(acc_list,histtype='bar', color=color,alpha=0.7)
    plt.xlabel('Classification Test Accuracy')
    plt.title('Histogram')
    plt.savefig(os.path.join(save_dir,f"{name}.jpg"))
    plt.close("all")

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    session_list = list(range(1, 4))
    people_list = list(range(1, 16))
    data_root_dir = os.path.join(os.getcwd(), "SEED-IV")

    alters = ["withinsubject", "betweensubject"]
    mode = alters[1]

    if mode == "withinsubject":
        for session in session_list:
            for people in people_list:
                X_train, y_train = load_one_subject(data_root_dir, str(session), str(people), True)
                X_test, y_test = load_one_subject(data_root_dir, str(session), str(people), False)
                print(X_train.shape)
                print(X_test.shape)
    else:
        for people in people_list:
            train_dataset = SEED_IV_DATASET(people=people, is_train=True, use_geometry=False)
            test_dataset = SEED_IV_DATASET(people=people, is_train=False, use_geometry=False)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
            print(train_dataloader)
            print(test_dataloader)
            print(X)

            X_train, y_train = load_all_sessions(data_root_dir, str(people), True)
            X_test, y_test = load_all_sessions(data_root_dir, str(people), False)
            print(X_train.shape)
            print(X_test.shape)

