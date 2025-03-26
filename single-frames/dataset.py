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

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

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


# class CustomDataset(Dataset):
#     def __init__(self, train, pkl_path, labels_csv, video_dir, transform=None, normalize=True, min_val=9.0, max_val=28.5):
 
#         if train:
#             pkl_train
#         else:
#             pkl test
 
#         diving_data, diving_mapping, labels = load_data(pkl_path, labels_csv)

#         image_paths, y_class, y_score, usage = prepare_data(diving_data, labels, diving_mapping, video_dir)

#         self.image_paths = []
#         self.bboxes = []
#         self.joints
#         self.pose
#         self.shape
#         self.trans
 
#         self.image_paths = image_paths 
#         # self.labels_class = labels_class
#         # self.labels_score = labels_score
#         self.transform = transform
#         self.normalize = normalize
#         self.min_val = min_val
#         self.max_val = max_val
        

#     def __getitem__(self, idx):
        
#         img_path = self.image_paths[idx]
#         bbox = self.bboxes[idx]

#         img_path = self.image_paths[idx]

#         img = cv2.imread(img_path)
#         img = Image.fromarray(img)
#         # cv2.imwrite('test.jpg', img)

#         # img = Image.open(img_path).convert('RGB') 

#         if self.transform:
#             img = self.transform(img)


#         img = img.detach().numpy().transpose((1,2,0))
#         cv2.imwrite('test.jpg', img*255)
#         img = img[0]

#         label_class = torch.tensor(self.labels_class[idx]).type(torch.long)
#         label_score = torch.tensor(self.labels_score[idx]).type(torch.float)

#         if self.normalize:
#             label_score = (label_score - self.min_val) / (self.max_val - self.min_val)

#         sample = (img, label_class, label_score)
#         return sample

#     def __len__(self):
#         return len(self.image_paths)



# def load_data(pkl_path='datasets/individual_diving.pkl', labels_csv='datasets/labels.csv'):
#     with open(pkl_path, 'rb') as f:
#         diving_data = pickle.load(f)

#     diving_mapping = {0: '305C', 1: '405B', 2: '205B', 3: '5152B', 4: '107B', 5: '305B', 6: '407C',7: '5253B', 8: '6243D', 9: '207C', 10: '626C'}

#     labels = pd.read_csv(labels_csv)

#     return diving_data, diving_mapping, labels

# def prepare_data(diving_data, labels, diving_mapping, video_dir='datasets/videos'):
#     """ Prepare data for multi-task learning:
#         input: diving data and labels
#         output: image paths and two different label arrays (classification and regression) """
    
#     image_paths = []  
#     labels_class = []
#     labels_score = []
#     usage = []
#     bboxes = []
    

#     usage_dict = {}
#     for idx, row in labels.iterrows():
#         usage_dict[row['Path']] = row['Usage']

#     for entry in diving_data:
#         img_path = entry['img_path']

#         full_img_path = os.path.join(video_dir, img_path)
        
#         img_folder = '/'.join(img_path.split('/')[:-1]) 

#         if img_folder not in usage_dict:
#             continue 

#         usage_info = usage_dict[img_folder] 

#         label_class = list(diving_mapping.keys())[list(diving_mapping.values()).index(entry['type'])]
#         label_score = entry['Scores']

#         if usage_info == 'train' or usage_info == 'test':
#             image_paths.append(full_img_path)
#             labels_class.append(label_class)
#             labels_score.append(label_score)
#             usage.append(usage_info)
#             bboxes.append(Base64)

#     return image_paths, labels_class, labels_score, usage



# def get_dataloaders(pkl_path='datasets/individual_diving.pkl', labels_csv='datasets/labels.csv', video_dir='datasets/videos', bs=64, augment=True):
#     """ Prepare train, val, & test dataloaders for multi-task learning """
#     mu, st = 0, 255

#     test_transform = transforms.Compose([ 
#         # transforms.Resize((80, 40)), 
#         transforms.TenCrop(256),
#         transforms.ToTensor(),
#         # transforms.Lambda(lambda crops: torch.stack(
#         #     [transforms.ToTensor()(crop) for crop in crops])),
#         transforms.Lambda(lambda tensors: torch.stack(
#             [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))])

#     if augment:
#         train_transform = transforms.Compose([
#             # transforms.Resize((256, 256)), 
#             transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),
#             transforms.RandomApply([transforms.ColorJitter(
#                 brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
#             transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
#             # transforms.FiveCrop(40),
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=(mu,), std=(st,))
#             # transforms.Lambda(lambda crops: torch.stack(
#             #     [transforms.ToTensor()(crop) for crop in crops])),
#             # transforms.Lambda(lambda tensors: torch.stack(
#             #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))
#             ]
#                                              )
#     else:
#         train_transform = test_transform


#     train_dataset = CustomDataset(True, pkl_path, labels_csv, video_dir, train_transform)
#     val_dataset = CustomDataset(False, pkl_path, labels_csv, video_dir, test_transform)

#     train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)

#     return train_loader, val_loader

# import os
# import cv2
# import numpy as np
# import pandas as pd
# from PIL import Image

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# class CustomDataset(Dataset):
#     def __init__(self, images, labels_class, labels_score, transform=None, augment=False):
#         self.images = images
#         self.labels_class = labels_class
#         self.labels_score = labels_score
#         self.transform = transform
#         self.augment = augment

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img = np.array(self.images[idx])
#         img = Image.fromarray(img)

#         if self.transform:
#             img = self.transform(img)

#         label_class = torch.tensor(self.labels_class[idx]).type(torch.long)
#         label_score = torch.tensor(self.labels_score[idx]).type(torch.float)

#         sample = (img, label_class, label_score)
#         return sample

# def load_data(path='datasets/diving/diving.csv'):
#     diving = pd.read_csv(path)
#     diving_mapping = {0: '305C', 1: '405B', 2: '205B', 3: '5152B', 4: '107B', 5: '305B', 6: '407C',7: '5253B', 8: '6243D', 9: '207C', 10: '626C'}

#     return diving, diving_mapping

# def prepare_data(data):
#     """ Prepare data for multi-task learning:
#         input: data frame with labels and pixel data
#         output: image and two different label arrays (classification and regression) """
    

#     image_array = np.zeros(shape=(len(data), 48, 48))


#     image_label_class = np.array(list(map(int, data['emotion'])))
    

#     image_label_score = np.array(list(map(float, data['score'])))  
    
#     for i, row in enumerate(data.index):
#         image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
#         image = np.reshape(image, (48, 48))
#         image_array[i] = image

#     return image_array, image_label_class, image_label_score

# def get_dataloaders(path='datasets/diving/diving.csv', bs=64, augment=True):
#     """ Prepare train, val, & test dataloaders for multi-task learning """
#     diving, diving_mapping = load_data(path)

#     # 准备数据
#     xtrain, ytrain_class, ytrain_score = prepare_data(diving[diving['Usage'] == 'Training'])
#     xval, yval_class, yval_score = prepare_data(diving[diving['Usage'] == 'PrivateTest'])
#     xtest, ytest_class, ytest_score = prepare_data(diving[diving['Usage'] == 'PublicTest'])

#     mu, st = 0, 255

#     test_transform = transforms.Compose([ 
#         transforms.Grayscale(),
#         transforms.TenCrop(40),
#         transforms.Lambda(lambda crops: torch.stack(
#             [transforms.ToTensor()(crop) for crop in crops])),
#         transforms.Lambda(lambda tensors: torch.stack(
#             [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))])

#     if augment:
#         train_transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
#             transforms.RandomApply([transforms.ColorJitter(
#                 brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
#             transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
#             transforms.FiveCrop(40),
#             transforms.Lambda(lambda crops: torch.stack(
#                 [transforms.ToTensor()(crop) for crop in crops])),
#             transforms.Lambda(lambda tensors: torch.stack(
#                 [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
#             transforms.Lambda(lambda tensors: torch.stack(
#                 [transforms.RandomErasing()(t) for t in tensors]))])
#     else:
#         train_transform = test_transform

#     # 创建数据集
#     train = CustomDataset(xtrain, ytrain_class, ytrain_score, train_transform)
#     val = CustomDataset(xval, yval_class, yval_score, test_transform)
#     test = CustomDataset(xtest, ytest_class, ytest_score, test_transform)

#     # 创建数据加载器
#     trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=4)
#     valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=4)
#     testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=4)

#     return trainloader, valloader, testloader


