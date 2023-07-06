#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   scripts.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/6 21:31     mql        1.0         None
'''
import numpy as np
import random
import os
import torch
import pickle
import pandas
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils import visualize_acc

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    dependence_path = os.path.join(os.getcwd(), "dependence")
    independence_path = os.path.join(os.getcwd(), "independence")
    test_obj = "/home/user/mql/my-project/NN_hw/Assignment3/dependece/model:1-lr:0.0001-epochs:3000/acc-3000.obj"
    test_path = "/".join(test_obj.split("/")[:-1]) + "/"

    acc_list = load_obj(test_obj)
    visualize_acc(acc_list=acc_list, save_dir=test_path, name="fig")


