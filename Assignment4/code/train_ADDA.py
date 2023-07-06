#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_ADDA.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/5 15:01     mql        1.0         None
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
from ADDA import Feature_Extractor, Classifier, Domain_Classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Training DANN")
    parser.add_argument(
        "--model",
        type=str,
        default="ADDA"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--epochs_pretrain",
        type=int,
        default=1
    )
    parser.add_argument(
        "--epochs_adapt",
        type=int,
        default=1
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

def pretrain(src_Mapper,label_Classififier,
             src_datalodaer, lr, epochs, device):
    src_Mapper.to(device)
    label_Classififier.to(device)
    optimizer = optim.AdamW([{'params': src_Mapper.parameters()},
                             {'params': label_Classififier.parameters()}],lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with tqdm(range(epochs), desc="pretraining Process") as pbar:
        for epoch in pbar:
            src_Mapper.train()
            label_Classififier.train()
            total_label_loss = 0.0
            for index, batch in enumerate(src_datalodaer):
                source_feature, source_class_labels = batch[0].to(device), batch[1].to(device)
                pred_label = label_Classififier(src_Mapper(source_feature))
                loss = loss_fn(pred_label, source_class_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_label_loss += loss.item()
            print("Summary Epoch {} training loss {}".format(epoch + 1, total_label_loss / len(src_datalodaer)))


def Adv_adaptation(tar_Mapper,domain_Classifier, src_Mapper,
             src_datalodaer, tar_dataloader, lr, epochs, device):
    optimizer_G = optim.SGD(tar_Mapper.parameters(), lr=lr)
    optimizer_D = optim.SGD(domain_Classifier.parameters(), lr=lr)

    tar_Mapper.to(device)
    src_Mapper.to(device)
    domain_Classifier.to(device)

    loss_fn = nn.CrossEntropyLoss()
    target_iter = iter(tar_dataloader)
    tar_Mapper.train()
    domain_Classifier.train()

    with tqdm(range(epochs), desc="Adversarial Adaptation Process") as pbar:
        for epoch in pbar:
            loss_G_all = 0.0
            loss_D_all = 0.0

            # train the Generator
            for index, batch in enumerate(src_datalodaer):
                source_feature, source_class_labels = batch[0].to(device), batch[1].to(device)
                try:
                    target_feature, target_class_labels = next(target_iter)
                except StopIteration:
                    target_iter = iter(tar_dataloader)
                    target_feature, target_class_labels = next(target_iter)

                target_feature, target_class_labels = target_feature.to(device), target_class_labels.to(device)
                source_domain_label = torch.zeros_like(source_class_labels, dtype=torch.long).to(device)
                target_domain_label = torch.ones_like(target_class_labels, dtype=torch.long).to(device)

                source_feature, target_feature = src_Mapper(source_feature), tar_Mapper(target_feature)
                feature_total = torch.cat([source_feature, target_feature], dim=0)
                total_domain_label = torch.cat([source_domain_label, target_domain_label],dim=0)
                loss_G = loss_fn(domain_Classifier(feature_total), 1 - total_domain_label)  # 1- is for confusing the domain cls

                loss_G_all += loss_G.item()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            # train the Discriminator
            for index, batch in enumerate(tar_dataloader):
                target_feature, target_class_labels = batch[0].to(device), batch[1].to(device)

                target_domain_label = torch.ones_like(target_class_labels, dtype=torch.long).to(device)

                target_feature = tar_Mapper(target_feature)

                loss_D = loss_fn(domain_Classifier(target_feature), target_domain_label)

                loss_D_all += loss_D.item()
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            print("Summary Epoch {} Generator loss {}  Discriminator loss {}".format(epoch + 1,loss_G_all / len(src_datalodaer), loss_D_all / len(tar_dataloader)))

def test_on_target(people, tar_Mapper, label_Classififier, tar_dataloader, device):
    n_correct = 0
    n_total = 0
    tar_Mapper.eval()
    label_Classififier.eval()
    acc_t_history = list()
    with torch.no_grad():
        for index, batch in enumerate(tar_dataloader):
            target_feature, target_class_labels = batch[0].to(device), batch[1].to(device)
            target_feature = tar_Mapper(target_feature)
            output_label = label_Classififier(target_feature)

            pred = output_label.detach().argmax(dim=1, keepdim=False)
            n_correct += pred.eq(target_class_labels.detach().view_as(pred)).cpu().sum()
            n_total += len(pred)

        acc_t = n_correct / n_total
        acc_t_history.append(acc_t)

    print('Summary subject: %d,  target accuracy: %.4f'
          % (people, sum(acc_t_history) / len(acc_t_history)))

    return sum(acc_t_history) / len(acc_t_history)

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

        src_Mapper = Feature_Extractor()
        tar_Mapper = Feature_Extractor()
        label_Classififier = Classifier()
        domain_Classifier = Domain_Classifier()

        #pretrian the source feature extractor
        print("Pretraining stage")
        pretrain(src_Mapper=src_Mapper, label_Classififier=label_Classififier, lr=args.lr,epochs=args.epochs_pretrain,
                 src_datalodaer=source_loader, device=device)
        for params in src_Mapper.parameters():
            params.requires_grad = False
        for params in label_Classififier.parameters():
            params.requires_grad = False

        #Adv adaptation Stage
        print("Adversarial Adaptation Stage")
        tar_Mapper.load_state_dict(src_Mapper.state_dict())
        src_Mapper.eval()
        label_Classififier.eval()
        Adv_adaptation(tar_Mapper=tar_Mapper, src_Mapper=src_Mapper, domain_Classifier=domain_Classifier,
                       tar_dataloader=target_loader, lr=args.lr,epochs=args.epochs_adapt, src_datalodaer=source_loader,
                       device=device)

        # test stage
        print("Testing Stage")
        acc_subject = test_on_target(people=people, tar_Mapper=tar_Mapper, label_Classififier=label_Classififier,
                       tar_dataloader=target_loader, device=device)
        acc_history.append(acc_subject)


    # save results into folder
    save_folder = os.path.join("crosssubject","ADDA",f"model:{args.model}-lr:{args.lr}-epochs_pretrain:{args.epochs_pretrain}-epochs_adapt:{args.epochs_adapt}")
    makedirs(save_folder)
    save_obj(acc_history, os.path.join(save_folder, f"acc-epochs_pretrain:{args.epochs_pretrain}-epochs_adapt:{args.epochs_adapt}.obj"))
    log_path = os.path.join(save_folder, "log.txt")
    with open(log_path, "w") as f:
        print(f"\n Average accuracy for all 15 models {sum(acc_history) / len(acc_history)}", file=f)
        print(f"\n Mean Accuracy: {np.mean(acc_history)}", file=f)
        print(f"\n Std Accuracy: {np.std(acc_history)}", file=f)
    visualize_acc(acc_history, save_folder, "fig")
    print("average accuracy for all 15 models {}".format(sum(acc_history) / len(acc_history)))

if __name__ == "__main__":
    args = parse_args()
    main(args)