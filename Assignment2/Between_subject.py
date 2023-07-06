#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Between_subject.py.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/19 0:10     mql        1.0         None
'''
import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from utils import *

def load_data_all_sessions(data_root_dir,people,
                is_train=True, session_list = [1, 2, 3]):
    data_all = []
    label_all = []
    for session in session_list:
        dir_list = os.listdir(os.path.join(data_root_dir, str(session)))
        for f in dir_list:
            if eval(f.split("_")[0]) == people:
                real_dir = os.path.join(data_root_dir, str(session), f)
                if is_train:
                    data = np.load(os.path.join(real_dir, "./train_data.npy"))
                    data = data.reshape(data.shape[0], -1)
                    label = np.load(os.path.join(real_dir, "./train_label.npy"))
                else:
                    data = np.load(os.path.join(real_dir, "./test_data.npy"))
                    data = data.reshape(data.shape[0], -1)
                    label = np.load(os.path.join(real_dir, "./test_label.npy"))
                data_all.append(data)
                label_all.append(label)
    data_all = np.concatenate(data_all,axis=0)
    label_all = np.concatenate(label_all,axis=0)
    return data_all, label_all


def train_one_vs_rest(category,
                      train_data, train_label,
                      test_data, test_label,
                      kernel, tol, regularization_parm, class_weight, max_iter):
    if class_weight == 'None':
        class_weight = None
    transformed_train_label = np.where(train_label == category, 1, 0)
    transformed_test_label = np.where(test_label == category, 1, 0)
    # clf = make_pipeline(StandardScaler(),
    #                     SVC(kernel=kernel, tol=tol, C=regularization_parm, class_weight=class_weight, probability=True))
    clf = SVC(kernel=kernel, tol=tol, C=regularization_parm, max_iter=max_iter, class_weight=class_weight, probability=True)

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    clf.fit(train_data_scaled, transformed_train_label)
    acc = clf.score(test_data_scaled, transformed_test_label)
    print("testing accuracy of SVM for category {}: %f".format(category) % acc)
    return clf


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default="./SEED-IV/", help="root dir of data")
    parser.add_argument("--kernel", type=str, default="rbf", help="kernel used in SVC")
    parser.add_argument("--tol", type=float, default=1e-3, help="tolerance")
    parser.add_argument("--regularization_parm", type=float, default=0.8, help="regularization_parm")
    parser.add_argument("--class_weight", type=None, default=None, help="weight of class, alternative: None(equal), balanced")
    parser.add_argument("--max_iter", type=int, default=10, help="iterations for training a model")
    args = parser.parse_args()

    set_seed(seed=42)

    session_list = list(range(1, 4))
    people_list = list(range(1, 16))

    pred_list = [[] for _ in session_list]

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    # cat the 3 sessions
    for people in people_list:
        X_train, y_train = load_data_all_sessions(args.data_root_dir, people, True, session_list)
        X_test, y_test = load_data_all_sessions(args.data_root_dir, people, False, session_list)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    acc_list = np.zeros(shape=(15))
    pred_list = []

    pbar = tqdm(people_list, mininterval=1, ncols=100)
    for people in pbar:
        X_train_others, y_train_others = X_train_list[:people - 1] + X_train_list[people:] + X_test_list[:people - 1] + \
                                         X_test_list[people:], y_train_list[:people - 1] + y_train_list[people:] + \
                                                               y_test_list[:people - 1] + y_test_list[people:]


        X_test_one, y_test_one = X_train_list[people - 1:people] + X_test_list[people - 1:people], \
            y_train_list[people - 1:people] + y_test_list[people - 1:people]


        X_train_others = np.concatenate(X_train_others, axis=0)
        y_train_others = np.concatenate(y_train_others, axis=0)
        X_test_one = np.concatenate(X_test_one, axis=0)
        y_test_one = np.concatenate(y_test_one, axis=0)

        clf0 = train_one_vs_rest(category=0, train_data=X_train_others, train_label=y_train_others,
                                 test_data=X_test_one, test_label=y_test_one,
                                 kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                 class_weight=args.class_weight, max_iter=args.max_iter)

        clf1 = train_one_vs_rest(category=1, train_data=X_train_others, train_label=y_train_others,
                                 test_data=X_test_one, test_label=y_test_one,
                                 kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                 class_weight=args.class_weight, max_iter=args.max_iter)

        clf2 = train_one_vs_rest(category=2, train_data=X_train_others, train_label=y_train_others,
                                 test_data=X_test_one, test_label=y_test_one,
                                 kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                 class_weight=args.class_weight, max_iter=args.max_iter)

        clf3 = train_one_vs_rest(category=3, train_data=X_train_others, train_label=y_train_others,
                                 test_data=X_test_one, test_label=y_test_one,
                                 kernel=args.kernel, tol=args.tol, regularization_parm=args.regularization_parm,
                                 class_weight=args.class_weight, max_iter=args.max_iter)
        prob0 = clf0.predict_proba(X_test_one)[:, 1]
        prob1 = clf1.predict_proba(X_test_one)[:, 1]
        prob2 = clf2.predict_proba(X_test_one)[:, 1]
        prob3 = clf3.predict_proba(X_test_one)[:, 1]

        test_pred = np.vstack((prob0, prob1, prob2, prob3))
        test_pred = np.argmax(test_pred, axis=0)
        accuracy = np.sum(test_pred == y_test_one) / len(y_test_one)
        acc_list[people - 1] = accuracy
        print("final accuracy of test subject {}: %f".format(people) % accuracy)

    print("final accuracy:", sum(acc_list) / len(acc_list))

    save_folder = os.path.join("independece", str(args.class_weight), f"{args.regularization_parm}-{args.max_iter}")
    makedirs(save_folder)
    save_obj(acc_list, os.path.join(save_folder, f"acc-{args.max_iter}.obj"))

    print(np.mean(acc_list))

    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Mean Accuracy: {np.mean(acc_list)}", file=f)
        print(f"\n Std Accuracy: {np.std(acc_list)}", file=f)

    visualize_acc(acc_list, save_folder, "fig")