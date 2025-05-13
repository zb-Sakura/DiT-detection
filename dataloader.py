import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, image_size=224, mode='train', augment=False):
        self.data_dir = data_dir
        self.mode = mode
        self.augment = augment
        self.image_size = image_size

        # 获取所有图像路径
        self.image_paths = []
        self.labels = []

        # 假设数据目录结构为: data_dir/healthy/*.png 和 data_dir/lesion/*.png
        healthy_dir = os.path.join(data_dir, 'healthy')
        lesion_dir = os.path.join(data_dir, 'lesion')

        # 添加健康图像
        if os.path.exists(healthy_dir):
            for img_name in os.listdir(healthy_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(healthy_dir, img_name))
                    self.labels.append(0)  # 0表示健康

        # 添加病变图像
        if os.path.exists(lesion_dir):
            for img_name in os.listdir(lesion_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(lesion_dir, img_name))
                    self.labels.append(1)  # 1表示病变

        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道医学图像
        ])

        # 数据增强（仅用于训练集）
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图像
        image = Image.open(img_path).convert('L')  # 转换为单通道

        # 应用变换
        image = self.transform(image)

        # 训练时进行数据增强
        if self.augment and self.mode == 'train':
            image = self.augment_transform(image)

        # 对于病变图像，我们还需要创建一个"目标健康图像"
        # 这里简单地使用健康图像的平均表示，实际应用中可能需要更复杂的方法
        target_label = 0  # 我们总是尝试生成健康图像

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'target_label': torch.tensor(target_label, dtype=torch.long)
        }


def get_dataloader(data_dir, image_size=224, batch_size=32, mode='train'):
    augment = (mode == 'train')
    dataset = MedicalImageDataset(data_dir, image_size, mode, augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )

    return dataloader