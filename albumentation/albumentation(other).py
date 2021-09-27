from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve

import albumentations as A
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import GaussianBlur, HorizontalFlip, HueSaturationValue, RandomShadow, RandomSnow
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
cudnn.benchmark = True

# 이미지 경로 
dataset_directory = os.path.join('.', 'SFR-03(normal)')
images_filepaths = sorted([os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory)])
images_filepaths = [*images_filepaths]
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
train_images_filepaths = correct_images_filepaths[:]

# val_images_filepaths = correct_images_filepaths[10:]

# BGR2RGB, Labeling
class CatsVsDogsDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "Cat":
            label = 1.0
        else:
            label = 0.0
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
        
# GaussianBlur sigma : 가우시안 커널의 표준 편차, sigma 값이 클수록 더 부드러워짐(노이즈 제거를 초점으로 한다면 대부분 0으로 설정)
# GassuanBlur 필터 크기가 커질수록, 평균을 내는 구간이 커지기 때문에, 더더욱 흐린 사진이 된다.
train_transform = A.Compose(
    [
        # A.Resize(256, 256, cv2.INTER_LINEAR),
        # A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5),
        # A.Affine(rotate=[-20, 20], interpolation=1, fit_output=False, always_apply=False, p=1),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2(),
    ]
)

train_dataset = CatsVsDogsDataset(images_filepaths=train_images_filepaths, transform=train_transform)


def write_image(dataset, idx=0):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    image, _ = dataset[idx]
    # print(dataset)
    # print(type(dataset))
    # print(len(dataset))
    for i in range(len(dataset)):
        image, _ = dataset[i]
        cv2.imwrite(f'albumentation/image(normal)/img_{i+516}.png',image) # 43개 단위

write_image(train_dataset)


# def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
#     dataset = copy.deepcopy(dataset)
#     dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
#     rows = samples // cols
#     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
#     for i in range(samples):
#         image, _ = dataset[idx]
#         ax.ravel()[i].imshow(image)
#         ax.ravel()[i].set_axis_off()
#     plt.tight_layout()
#     plt.show()

# random.seed(42)
# visualize_augmentations(train_dataset)



# val_transform = A.Compose(
#     [
        # A.Scale((224, 224)),
# A.Fliplr(0.5),
# A.Sometimes(0.25, A.GaussianBlur(sigma=(0, 3.0))),
# A.AddToHueAndSaturation(value=(-10, 10), per_channel=True
# A.Affine(rotate=(-20, 20), mode='symmetric'),
# A.Sometimes(0.25,
# A.OneOf([A.Dropout(p=(0, 0.1)),
# A.CoarseDropout(0.1, size_percent=0.5)])),
# A.SmallestMaxSize(max_size=160),
# A.RandomCrop(height=128, width=128),
#         A.SmallestMaxSize(max_size=160),
#         A.CenterCrop(height=128, width=128),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )
# val_dataset = CatsVsDogsDataset(images_filepaths=val_images_filepaths, transform=val_transform)