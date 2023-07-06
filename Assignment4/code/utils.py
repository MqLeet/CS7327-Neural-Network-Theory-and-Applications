#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/29 19:46     mql        1.0         None
'''
import numpy as np
import random
import pickle
import os
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def save_obj(obj ,path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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

