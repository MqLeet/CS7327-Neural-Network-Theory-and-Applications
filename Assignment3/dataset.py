from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch
import numpy as np
import os

class SEEDIVDataset(Dataset):
    def __init__(self, dir_path, split=0):
        super(SEEDIVDataset, self).__init__()
        if split==0:
            x_train = np.load(os.path.join(dir_path, 'train_data.npy'))
            train_label = np.load(os.path.join(dir_path, 'train_label.npy'))
        else:
            x_train = np.load(os.path.join(dir_path, 'test_data.npy'))
            train_label = np.load(os.path.join(dir_path, 'test_label.npy'))
        self.data_size = x_train.shape[0]
        x_train_new = torch.zeros(self.data_size, 81, 5)
        x_train = torch.from_numpy(x_train)
        pos_list = [2, 4, 6, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 69, 70, 71, 73, 74, 76, 78, 79]
        x_train_new[:, pos_list, :] = x_train.float()
        self.x = (x_train_new.reshape(-1, 9, 9, 5).permute(0,3,1,2))/20.0
        
        self.y = torch.from_numpy(train_label.astype('int64'))
        # train_label=torch.from_numpy(train_label.astype('int64')).unsqueeze(dim=1)
        # self.y = torch.zeros(610,4).scatter_(1,train_label,1)
        # self.y[]
                    # print(y_predict)
    def __len__(self):
        return self.data_size
    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        return img, label

class SEEDIVDataset2(Dataset):
    def __init__(self, x_train, train_label):
        super(SEEDIVDataset2, self).__init__()
        self.data_size = x_train.shape[0]
        x_train_new = torch.zeros(self.data_size, 81, 5)
        x_train = torch.from_numpy(x_train)
        pos_list = [2, 4, 6, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 69, 70, 71, 73, 74, 76, 78, 79]
        x_train_new[:, pos_list, :] = x_train.float()
        self.x = (x_train_new.reshape(-1, 9, 9, 5).permute(0,3,1,2))/20.0
        
        self.y = torch.from_numpy(train_label.astype('int64'))
    def __len__(self):
        return self.data_size
    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        return img, label