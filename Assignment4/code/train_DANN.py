#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_DANN.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/28 15:46     mql        1.0         None
'''
import torch
import torch.optim as optim
from tqdm import tqdm
import os, sys
import argparse
from dataset import SEED_IV_DATASET
from torch.utils.data import DataLoader
from utils import set_seed, makedirs, save_obj, visualize_acc
from DANN import DANN
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Training DANN")
    parser.add_argument(
        "--model",
        type=str,
        default="DANN"
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
        "--_lambda",
        type=float,
        default=1.0
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
    parser.add_argument(
        "--label_classifier_hidden_dim",
        type=int,
        default=128
    )
    parser.add_argument(
        "--domain_classifier_hidden_dim",
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
        model = DANN(label_classifier_hidden_dim=args.label_classifier_hidden_dim,
                     domain_classifier_hidden_dim=args.domain_classifier_hidden_dim)

        acc_subject = train_DANN(model=model, subject=people, lr=args.lr, epochs=args.epochs,
                   source_loader=source_loader, target_loader=target_loader, device=device)

        acc_history.append(acc_subject)

    # save results into folder
    save_folder = os.path.join("crosssubject", "DANN",f"model:{args.model}-lr:{args.lr}-epochs:{args.epochs}-lambda:{args._lambda}")
    makedirs(save_folder)
    save_obj(acc_history, os.path.join(save_folder, f"acc-{args.epochs}.obj"))
    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Average accuracy for all 15 models {sum(acc_history) / len(acc_history)}", file=f)
        print(f"\n Mean Accuracy: {np.mean(acc_history)}", file=f)
        print(f"\n Std Accuracy: {np.std(acc_history)}", file=f)
    visualize_acc(acc_history, save_folder, "fig")
    print("average accuracy for all 15 models {}".format(sum(acc_history) / len(acc_history)))




def train_DANN(model, subject, lr, epochs,
               source_loader, target_loader, device, _lambda=1.0):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_label = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    acc_history = []
    target_iter = iter(target_loader)

    with tqdm(range(epochs), desc="Training Process for test_people {}".format(subject)) as pbar:
        for epoch in pbar:
            model.train()
            total_label_loss = 0.0
            total_domain_loss = 0.0

            # training model use source domain
            for index, batch in enumerate(source_loader):
                source_feature, source_class_labels = batch[0].to(device), batch[1].to(device)
                source_domain_labels = torch.zeros_like(source_class_labels, dtype=torch.long).to(device)

                try:
                    target_feature, target_class_labels = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_feature, target_class_labels = next(target_iter)

                target_feature, target_class_labels = target_feature.to(device), target_class_labels.to(device)
                target_domain_labels = torch.ones_like(target_class_labels, dtype=torch.long).to(device)

                p = float(index + epoch * len(source_loader)) / epochs / len(source_loader)
                _lambda = 2. / (1. + np.exp(-10 * p)) - 1


                output_label, output_domain = model(source_feature, _lambda)
                loss_s_label = loss_label(output_label, source_class_labels)
                loss_s_domain = loss_domain(output_domain, source_domain_labels)

                output_label, output_domain = model(target_feature, _lambda)
                loss_t_domain = loss_domain(output_domain, target_domain_labels)

                optimizer.zero_grad()
                loss = loss_s_label + loss_s_domain + loss_t_domain
                loss.backward()
                optimizer.step()

                total_label_loss += loss_s_label.detach().cpu().numpy()
                total_domain_loss += loss_s_domain.detach().cpu().numpy() + loss_t_domain.detach().cpu().numpy()


            n_correct = 0
            n_total = 0
            model.eval()
            acc_t_history = list()
            with torch.no_grad():
                for index, batch in enumerate(target_loader):
                    target_feature, target_class_labels = batch[0].to(device), batch[1].to(device)
                    output_label, _ = model(target_feature, _lambda)


                    pred = output_label.detach().argmax(dim=1, keepdim=False)
                    n_correct += pred.eq(target_class_labels.detach().view_as(pred)).cpu().sum()
                    n_total += len(pred)

                acc_t = n_correct / n_total
                acc_t_history.append(acc_t)

            print('Summary Epoch: %d, loss_label: %.4f, loss_domain: %.4f, target accuracy: %.4f'
                  % (epoch, total_label_loss / len(source_loader), total_domain_loss / len(source_loader), sum(acc_t_history) / len(acc_t_history)))
            acc_history.append(sum(acc_t_history) / len(acc_t_history))

    return max(acc_history)

if __name__ == "__main__":
    args = parse_args()
    main(args)