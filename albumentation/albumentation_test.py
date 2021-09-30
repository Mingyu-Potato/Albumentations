import copy
import os
import numpy as np
import albumentations as A
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import GaussianBlur, HorizontalFlip, HueSaturationValue, RandomShadow, RandomSnow
from albumentations.pytorch import ToTensorV2
import cv2

image = cv2.imread('./image/SFR-03(Normal)/origin/017A8500.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

albumentation_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(rotate=[-20, 20], interpolation=1, fit_output=False, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
    ]
)

augmented = albumentation_transform(image=image)
image = augmented['image']
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
print(type(image))
cv2.imwrite(f'./image.png', image)