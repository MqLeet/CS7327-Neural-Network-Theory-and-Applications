#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_baseline.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/5 19:18     mql        1.0         None
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os, sys
import argparse
from dataset import SEED_IV_DATASET
from torch.utils.data import DataLoader
from utils import set_seed, makedirs, save_obj, visualize_acc
from DANN import BaseNet

def parse_args():
    parser = argparse.ArgumentParser(description="Training DANN")
    parser.add_argument(
        "--model",
        type=str,
        default="Baseline"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )
    return parser.parse_args()



def main(args):
    set_seed(seed=2023)
    device = args.device
    people_list = list(range(1, 16))
    acc_history = []

    for people in people_list:
        train_dataset = SEED_IV_DATASET(people=people, is_train=True, use_geometry=True)
        test_dataset = SEED_IV_DATASET(people=people, is_train=False, use_geometry=True)
        source_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        target_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
        model = BaseNet()

        acc_subject = train_BaseNet(model=model, subject=people, lr=args.lr, epochs=args.epochs,
                   source_loader=source_loader, target_loader=target_loader, device=device)

        acc_history.append(acc_subject)

    # save results into folder
    save_folder = os.path.join("crosssubject", "Baseline",f"model:{args.model}-lr:{args.lr}-epochs:{args.epochs}")
    makedirs(save_folder)
    save_obj(acc_history, os.path.join(save_folder, f"acc-{args.epochs}.obj"))
    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Average accuracy for all 15 models {sum(acc_history) / len(acc_history)}", file=f)
        print(f"\n Mean Accuracy: {np.mean(acc_history)}", file=f)
        print(f"\n Std Accuracy: {np.std(acc_history)}", file=f)
    visualize_acc(acc_history, save_folder, "fig")
    print("average accuracy for all 15 models {}".format(sum(acc_history) / len(acc_history)))


def train_BaseNet(model, subject, lr, epochs,
               source_loader, target_loader, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc_history = []
    with tqdm(range(epochs), desc="Training Process for test_people {}".format(subject)) as pbar:
        for epoch in pbar:
            model.train()
            total_label_loss = 0.0


            # training model use source domain
            for index, batch in enumerate(source_loader):
                source_feature, source_class_labels = batch[0].to(device), batch[1].to(device)

                output_label = model(source_feature)
                loss_label = loss_fn(output_label, source_class_labels)
                optimizer.zero_grad()
                loss_label.backward()
                optimizer.step()
                total_label_loss += loss_label.detach().cpu().numpy()

            # testing model use target domain
            n_correct = 0
            n_total = 0
            model.eval()
            acc_t_history = list()
            with torch.no_grad():
                for index, batch in enumerate(target_loader):
                    target_feature, target_class_labels = batch[0].to(device), batch[1].to(device)
                    output_label= model(target_feature)


                    pred = output_label.detach().argmax(dim=1, keepdim=False)
                    n_correct += pred.eq(target_class_labels.detach().view_as(pred)).cpu().sum()
                    n_total += len(pred)

                acc_t = n_correct / n_total
                acc_t_history.append(acc_t)

            print('Summary Epoch: %d, loss_label: %.4f, target accuracy: %.4f'
                  % (epoch, total_label_loss / len(source_loader), sum(acc_t_history) / len(acc_t_history)))
            acc_history.append(sum(acc_t_history) / len(acc_t_history))

    return max(acc_history)



if __name__ == "__main__":
    args = parse_args()
    main(args)