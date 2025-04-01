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
            self.df['Scores'] = self.scores
            
        self.transform = transform

        self.df['video_folder'] = self.df['img_path'].apply(lambda x: os.path.dirname(x))
        self.df['frame_number'] = self.df['img_path'].apply(
            lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        self.videos = []
        for folder, group in self.df.groupby('video_folder'):
            group_sorted = group.sort_values('frame_number')
            self.videos.append(group_sorted)
    
    def _map_class_to_number(self, label):
        return reverse_diving_mapping.get(label, -1)
    
    def __getitem__(self, idx):
        video_group = self.videos[idx]
        img_paths = video_group['img_path'].tolist()

        imgs = [Image.open(p).convert('RGB') for p in img_paths]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
         
        video_tensor = torch.stack(imgs, dim=0)
        
        # debug_folder = "debug_frames"
        # if not os.path.exists(debug_folder):
        #     os.makedirs(debug_folder)
        
        # for i, frame in enumerate(video_tensor):
        #     frame_np = frame.cpu().detach().numpy().transpose(1, 2, 0)
        #     frame_np = (frame_np * 255).astype(np.uint8)
        #     frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        #     save_path = os.path.join(debug_folder, f"video_{idx}_frame_{i}.jpg")
        #     cv2.imwrite(save_path, frame_np)
            
            
        label_class = torch.tensor(self._map_class_to_number(video_group.iloc[0]['type']), dtype=torch.long)
        score = torch.tensor(video_group.iloc[0]['Scores'], dtype=torch.float)

        return video_tensor, label_class, score
    
    def __len__(self):
        return len(self.videos)

def get_dataloaders(train_pkl='datasets/train.pkl', val_pkl='datasets/val.pkl', bs=1, augment=True):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    
    test_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = test_transform
    
    train_dataset = CustomDataset(pkl_path=train_pkl, transform=train_transform)
    val_dataset = CustomDataset(pkl_path=val_pkl, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

