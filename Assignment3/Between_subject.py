#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Between_subject.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/6 14:04     mql        1.0         None
'''
import numpy as np
import os, sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from utils import set_seed, load_one_subject, makedirs, save_obj, visualize_acc, info_transform, SEED_IV_DATASET
from models import BaseNet, AdvancedNet
os.environ['CUDA_VISIBLE_DEVICE']='0,1'
def train(model, lr, epochs, train_loader, test_loader, subject):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with tqdm(range(epochs), desc="Training Process for test_people {}".format(subject)) as pbar:
        train_loss = []
        valid_loss = []
        acc_list = []
        for epoch in pbar:
            running_loss = 0.0
            model.train()
            for index, batch in tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout,
                                     desc="Loading a batch of train_data"):
                inputs, labels = batch['data'], batch['label']
                inputs, labels = inputs.to(device), labels.to(device)
                if args.model == 0:
                    inputs = torch.unsqueeze(inputs, dim=1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print('Training Epoch [{}/{}], training Loss: {:.4f}'.format(epoch + 1, epochs,
                                                                             running_loss / (epoch + 1)))
                train_loss.append(running_loss / (epoch + 1))

                correct = 0
                total = 0
                with torch.no_grad():
                    for index, batch in tqdm(enumerate(test_loader), total=len(test_loader),
                                             file=sys.stdout,
                                             desc="Loading a batch of test_data"):
                        inputs, labels = batch['data'], batch['label']
                        inputs, labels = inputs.to(device), labels.to(device)
                        if args.model == 0:
                            inputs = torch.unsqueeze(inputs, dim=1)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss = loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    acc_list.append(correct / total)
                    print(print(
                        'Epoch: {} Accuracy of the network on the test set:{}'.format(epoch + 1, correct / total)))
                    print('Validing Epoch [{}/{}], validing Loss: {:.4f}'.format(epoch + 1, epochs, val_loss))
                    valid_loss.append(val_loss)
    return sum(acc_list) / len(acc_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default=os.path.join(os.getcwd(), "SEED-IV"), help="dir of SEED-IV")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=int, default=1, help="0: seen the feature as 1 * 62 * 5 figure, one conv"
                                                             "1: seen the feature as 9 * 9 * 5 figure")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    device = args.device
    set_seed(seed=42)

    people_list = list(range(1, 16))
    all_acc_list = []


    for people in people_list:
        # use a new model for each person
        if args.model == 0:
            train_dataset = SEED_IV_DATASET(people=people, is_train=True, use_geometry=False)
            test_dataset = SEED_IV_DATASET(people=people, is_train=False, use_geometry=False)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
            net = BaseNet()
            net = net.to(device)
            acc = train(model=net, subject=people, lr=args.lr, epochs=args.epochs,
                        train_loader=train_loader, test_loader=test_loader)
            all_acc_list.append(acc)
        elif args.model == 1:
            train_dataset = SEED_IV_DATASET(people=people, is_train=True, use_geometry=True)
            test_dataset = SEED_IV_DATASET(people=people, is_train=False, use_geometry=True)
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
            net = AdvancedNet()
            net = net.to(device)


            acc = train(model=net, subject=people, lr=args.lr, epochs=args.epochs,
                        train_loader=train_loader, test_loader=test_loader)
            all_acc_list.append(acc)

    # save results into folder
    save_folder = os.path.join("independece", f"model:{args.model}-lr:{args.lr}-epochs:{args.epochs}")
    makedirs(save_folder)
    save_obj(all_acc_list, os.path.join(save_folder, f"acc-{args.epochs}.obj"))
    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Mean Accuracy: {np.mean(all_acc_list)}", file=f)
        print(f"\n Std Accuracy: {np.std(all_acc_list)}", file=f)
    visualize_acc(all_acc_list, save_folder, "fig")
    print("average accuracy for all 15 models {}".format(sum(all_acc_list) / len(all_acc_list)))
