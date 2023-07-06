#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ADDA.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/5 11:47     mql        1.0         None
'''
import torch.nn as nn

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, 1),  # b*32*9*9
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # b*64*5*5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # b*128*3*3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 0),  # b*256*1*1
        )
    def forward(self, x):
        return self.backbone(x).squeeze(-1).squeeze(-1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cls_header = nn.Sequential(
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32,4),
        )
    def forward(self, x):
        return self.cls_header(x)

class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.cls_header = nn.Sequential(
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32,2),
        )

    def forward(self, x):
        return self.cls_header(x)

