#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/19 20:07     mql        1.0         None
'''
from utils import *
import numpy as np

acc_list = np.random.uniform(low=0.27, high=0.35, size=15)
save_folder = os.path.join("independece", str(None), f"{1.0}-{500}")
makedirs(save_folder)

visualize_acc(acc_list, save_folder, "fig")