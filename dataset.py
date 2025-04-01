import os
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from utils1.module_utils import vis_img

diving_mapping = {
    0: '305C', 1: '405B', 2: '205B', 3: '5152B', 4: '107B', 
    5: '305B', 6: '407C', 7: '5253B', 8: '6243D', 9: '207C', 10: '626C'
}

reverse_diving_mapping = {v: k for k, v in diving_mapping.items()}

class CustomDataset(Dataset):
    def __init__(self, pkl_path, video_dir='datasets/video', transform=None, normalize=True, min_val=9.0, max_val=28.5):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        self.df = pd.DataFrame(self.data)
        self.df['img_path'] = video_dir + '/' + self.df['img_path']

        self.bboxes = np.array(self.df['bbox'].tolist())
        self.joints = np.array(self.df['joints3d'].tolist())
        self.pose = np.array(self.df['pose'].tolist())
        self.shape = np.array(self.df['betas'].tolist())
        self.trans = np.array(self.df['trans'].tolist())
        self.camparams = np.array(self.df['camparams'].tolist())
        self.labels_class = np.array([self._map_class_to_number(label) for label in self.df['type']])
        self.scores = np.array(self.df['Scores'], dtype=np.float32)
        
        if normalize:
            self.scores = (self.scores - min_val) / (max_val - min_val)

        self.transform = transform

    def _map_class_to_number(self, label):
        return reverse_diving_mapping.get(label, -1)

    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        bbox = self.bboxes[idx]
        joints = self.joints[idx]
        pose = self.pose[idx]
        shape = self.shape[idx]
        trans = self.trans[idx]
        camparams = self.camparams[idx]
        score = torch.tensor(self.scores[idx], dtype=torch.float)
        label_class = torch.tensor(self.labels_class[idx], dtype=torch.long)

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        # img = img.detach().numpy().transpose((1,2,0))
         # cv2.imwrite('test.jpg', img*255)
         # img = img[0]

        return img, label_class, score

    def __len__(self):
        return len(self.df)

def get_dataloaders(train_pkl='datasets/train.pkl', val_pkl='datasets/val.pkl', bs=64, augment=True):
    """ Prepare train & val dataloaders """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    # test_transform = transforms.Compose([ 
        # transforms.Resize((80, 40)), 
        # transforms.TenCrop(256),
        # transforms.ToTensor(),
        # transforms.Lambda(lambda crops: torch.stack(
        #     [transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda tensors: torch.stack(
        #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))])
    test_transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if augment:
        train_transform = transforms.Compose([
            # transforms.Resize((256, 256)), 
            # transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            # transforms.FiveCrop(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))
            ]
                                             )
    else:
        train_transform = test_transform

    train_dataset = CustomDataset(pkl_path=train_pkl, transform=train_transform)
    val_dataset = CustomDataset(pkl_path=val_pkl, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,num_workers=2,pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,num_workers=2,pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

