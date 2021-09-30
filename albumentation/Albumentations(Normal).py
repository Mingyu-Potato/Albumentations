import os
import copy
from urllib.request import urlretrieve
import albumentations as A
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.augmentations.transforms import ColorJitter
import cv2
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# 이미지 경로 
dataset_directory = os.path.join('./image/SFR-03(Normal)', 'origin')
images_filepaths = sorted([os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory)])
images_filepaths = [*images_filepaths]
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
train_images_filepaths = correct_images_filepaths[:]


# augmentation 방법
albumentation_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3,7), sigma_limit=0, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0, always_apply=False, p=0.5),
        A.Affine(rotate=[-15, 15], interpolation=1, fit_output=False, always_apply=False, p=0.5),
    ]
)

# 이미지 저장
def write_image(dataset):
    dataset = copy.deepcopy(dataset)
    for i in range(len(dataset)):
        image = cv2.imread(dataset[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        albumentation_image = albumentation_transform(image=image)['image']
        albumentation_image = cv2.cvtColor(albumentation_image, cv2.COLOR_RGB2BGR)

        filename, file_extension = os.path.splitext(dataset[i])
        cv2.imwrite(f'./image/SFR-03(Normal)/SFR-03(Normal)/{filename[-8:]}_2.jpg', albumentation_image)

write_image(train_images_filepaths)