#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DANN.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/27 22:05     mql        1.0         None
'''
import torch.autograd
import torch.nn as nn


class ReverseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx._lambda # -lambda * grad
        return output, None


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
        return self.backbone(x)

class Label_Classifier(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(Label_Classifier, self).__init__()

        self.cls_header = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.cls_header(x)


class Domain_Classifier(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(Domain_Classifier, self).__init__()
        self.cls_header = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.cls_header(x)


class DANN(nn.Module):
    def __init__(self,
                 label_classifier_hidden_dim,
                 domain_classifier_hidden_dim):
        super(DANN, self).__init__()

        if label_classifier_hidden_dim is None:
            label_classifier_hidden_dim = 128

        if domain_classifier_hidden_dim is None:
            domain_classifier_hidden_dim = 128

        self.feature_extractor = Feature_Extractor()
        self.label_classifier = Label_Classifier(in_dim=256, hidden_dim=label_classifier_hidden_dim, out_dim=4)
        self.domain_classifier = Domain_Classifier(in_dim=256, hidden_dim=domain_classifier_hidden_dim, out_dim=2)

    def grad_reverse(self, x, _lambda):
        return ReverseGrad.apply(x, _lambda)

    def forward(self, x, _lambda):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, feature.shape[1])
        feature_reverse = self.grad_reverse(feature, _lambda)
        output_label = self.label_classifier(feature)
        output_domain = self.domain_classifier(feature_reverse)
        return output_label, output_domain




class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.feature_extractor = Feature_Extractor()
        self.label_classifier = Label_Classifier(in_dim=256, hidden_dim=128, out_dim=4)

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature.view(-1, feature.shape[1])
        prediciton = self.label_classifier(feature)
        return prediciton
