from PIL import Image
import math
import numpy as np
import random
import os
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NaturalDisasterDataset(Dataset):
    def __init__(self, root, mean, std, phase='train'):
        super().__init__()

        self.root = root
        self.phase = phase

        self.mean, self.std = mean, std

    def __getitem__(self, index):
        leading_zeros = ['0' for _ in range(4 - len(str(index)))]
        leading_zeros += list(str(index))
        img_name_num = ''.join(leading_zeros)

        x_path = os.path.join(self.root, 'images', f'train_{img_name_num}.png')
        y_path = os.path.join(self.root, 'gt', f'train_{img_name_num}.png')

        x = Image.open(x_path)
        y = Image.open(y_path).convert('L')

        return self.apply_transforms(x, y)

    
    def __len__(self):
        return len(os.listdir(os.path.join(self.root, 'images')))

    def apply_transforms(self, x,y):
        if self.phase == 'train':
            x_transforms = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(3, sigma=(0.1, 10)),
                transforms.ColorJitter(0.1,0.1,0.1,0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            y_transforms = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        else:
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            y_transforms = None

        seed = np.random.randint(23412304981723094)

        random.seed(seed)
        torch.manual_seed(seed)
        x = x_transforms(x)

        if y_transforms:
            random.seed(seed)
            torch.manual_seed(seed)
            y = y_transforms(y)
            
        y = torch.Tensor(np.array(y)).squeeze().long()
        return x, y


def get_data_loaders(root_dir, validation_split, batch_size):
    images_ds = NaturalDisasterDataset(
        root=root_dir, 
        mean=[0.46077183, 0.45584197, 0.41929824], 
        std=[0.18551224, 0.17078055, 0.17699541]
    )

    ds_lengths = [
        math.floor((1-validation_split) * len(images_ds)),
        math.ceil((validation_split *  len(images_ds)))
    ]
    train_ds, val_ds = random_split(images_ds,  ds_lengths)

    train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    return train_data_loader, val_data_loader