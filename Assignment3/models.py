#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/31 15:29     mql        1.0         None
'''
import argparse
import os
import torch.optim as optim
import torch
import torch.nn as nn
from utils import load_one_subject, set_seed


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 30 * 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 30 * 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, 1),  # 32*9*9
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),  # 64*9*9
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 64*5*5
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128*3*3
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 0),  # 256*1*1
        )
        self.Linear = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        cnn_result = self.cnn(x).squeeze(-1).squeeze(-1)
        result = self.Linear(cnn_result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default=os.path.join(os.getcwd(), "SEED-IV"), help="dir of SEED-IV")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    device = args.device
    set_seed(seed=42)

    # Initialize the model and optimizer
    net = BaseNet()
    net.to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trainloader = load_one_subject(data_root_dir=args.data_root_dir, session=1, subject=1, is_train=True)
    testloader = load_one_subject(data_root_dir=args.data_root_dir, session=1, subject=1, is_train=False)

    # Train the network
    for epoch in range(args.epochs):
        running_loss = 0.0
        inputs, labels = trainloader
        inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.unsqueeze(inputs, dim=1)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, running_loss / (epoch + 1)))
            print(inputs.shape[0])
            # print(shape)

    # Test the network
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, labels = testloader
        inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.unsqueeze(inputs, dim=1)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


