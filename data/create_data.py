import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import random
import numpy as np


from PIL import Image as m

def transform(image, mask, img_size, lr_img_size, is_crop=True):

    if is_crop:
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(img_size, img_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

    # 双线性插值成B的大小
    lr_image = image.resize((lr_img_size, lr_img_size), resample=m.BICUBIC)

    mask = np.array(mask).astype(np.long)

    nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Transform to tensor
    image = TF.to_tensor(image)
    image = nomal_fun_image(image)
    lr_image = TF.to_tensor(lr_image)
    lr_image = nomal_fun_image(lr_image)

    mask = TF.to_tensor(mask)

    return image, mask, lr_image

class GenerateData(data.Dataset):
    def __init__(self, image_dir, img_size, lr_img_size, augment=None):
        self.image_dir = image_dir + "images"
        self.label_dir = image_dir + "labels"
        self.image_paths = sorted([os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, x) for x in os.listdir(self.label_dir)])

        self.augment = augment
        self.img_size = img_size
        self.lr_img_size = lr_img_size

        self.length = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        img = m.open(image_path).convert('RGB')
        label = m.open(label_path).convert('L')
        if self.augment:
            img, label, img_lr = transform(img, label, self.img_size, self.lr_img_size)
        else:
            img, label, img_lr = transform(img, label, self.img_size, self.lr_img_size, is_crop=False)

        return img, label, img_lr
    def __len__(self):
        return self.length

