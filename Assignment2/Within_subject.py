#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Within_subject.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/18 20:27     mql        1.0         None
'''
import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from utils import *


def train_one_vs_rest(category,
                      train_data, train_label,
                      test_data, test_label,
                      kernel, tol, regularization_parm, class_weight):
    if class_weight == 'None':
        class_weight = None
    transformed_train_label = np.where(train_label == category, 1, 0)
    transformed_test_label = np.where(test_label == category, 1, 0)
    # clf = make_pipeline(StandardScaler(),
    #                     SVC(kernel=kernel, tol=tol, C=regularization_parm, class_weight=class_weight, probability=True))

    clf = SVC(kernel=kernel, tol=tol, C=regularization_parm, class_weight=class_weight, probability=True)
    clf.fit(train_data, transformed_train_label)
    acc = clf.score(test_data, transformed_test_label)
    print("testing accuracy of SVM for category {}: %f".format(category) % acc)
    return clf


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default="./SEED-IV/", help="root dir of data")
    parser.add_argument("--kernel", type=str, default="rbf", help="kernel used in SVC")
    parser.add_argument("--tol", type=float, default=1e-3, help="tolerance")
    parser.add_argument("--regularization_parm", type=float, default=0.8, help="regularization_parm")
    parser.add_argument("--class_weight", type=str, default=None, help="weight of class, alternative: None(equal), balanced")
    args = parser.parse_args()

    set_seed(seed=42)

    acc = []
    unfolders = os.listdir(args.data_root_dir)
    for unfolder in unfolders:
        subjects = os.listdir(os.path.join(args.data_root_dir, unfolder))
        for subject in subjects:
            train_data = np.load(os.path.join(args.data_root_dir, unfolder, subject, "train_data.npy"))
            train_label = np.load(os.path.join(args.data_root_dir, unfolder, subject, "train_label.npy"))
            test_data = np.load(os.path.join(args.data_root_dir, unfolder, subject, "test_data.npy"))
            test_label = np.load(os.path.join(args.data_root_dir, unfolder, subject, "test_label.npy"))

            train_data = train_data.reshape(train_data.shape[0], -1)
            test_data = test_data.reshape(test_data.shape[0], -1)

            clf0 = train_one_vs_rest(category=0, train_data=train_data, train_label=train_label,
                                        test_label=test_label, test_data=test_data,
                                        kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm, class_weight=args.class_weight)
            clf1 = train_one_vs_rest(category=1, train_data=train_data, train_label=train_label,
                                     test_label=test_label, test_data=test_data,
                                     kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                     class_weight=args.class_weight)
            clf2 = train_one_vs_rest(category=2, train_data=train_data, train_label=train_label,
                                     test_label=test_label, test_data=test_data,
                                     kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                     class_weight=args.class_weight)
            clf3 = train_one_vs_rest(category=3, train_data=train_data, train_label=train_label,
                                     test_label=test_label, test_data=test_data,
                                     kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                     class_weight=args.class_weight)

            prob0 = clf0.predict_proba(test_data)[:, 1]
            prob1 = clf1.predict_proba(test_data)[:, 1]
            prob2 = clf2.predict_proba(test_data)[:, 1]
            prob3 = clf3.predict_proba(test_data)[:, 1]

            test_pred = np.vstack((prob0, prob1, prob2, prob3))
            test_pred = np.argmax(test_pred, axis=0)
            accuracy = np.sum(test_pred == test_label) / len(test_label)
            acc.append(accuracy)
            print("final accuracy of subject {}: %f".format(subject) %accuracy)
    print("final accuracy:", sum(acc) / len(acc))

    save_folder = os.path.join("dependece", str(args.class_weight), f"{args.regularization_parm}", str(args.kernel))
    makedirs(save_folder)
    save_obj(acc, os.path.join(save_folder, f"acc-{args.regularization_parm}.obj"))

    print(np.mean(acc))

    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Mean Accuracy: {np.mean(acc)}", file=f)
        print(f"\n Std Accuracy: {np.std(acc)}", file=f)

    visualize_acc(acc, save_folder, "fig")














