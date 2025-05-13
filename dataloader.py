import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import nibabel as nib
import torchvision.transforms as transforms


class BRATS2020Dataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', augment=False):
        self.data_dir = data_dir
        self.mode = mode
        self.augment = augment
        self.image_size = image_size

        self.image_paths = []
        self.labels = []

        # 检查data_dir是否已经是训练或验证目录
        if mode == 'train' and os.path.basename(data_dir) == 'train':
            train_dir = data_dir
        elif mode == 'validation' and os.path.basename(data_dir) == 'validation':
            train_dir = data_dir
        else:
            # 否则在data_dir下查找子目录
            sub_dir = 'train' if mode == 'train' else 'validation'
            train_dir = os.path.join(data_dir, sub_dir)

        if not os.path.exists(train_dir):
            raise ValueError(f"目录不存在: {train_dir}")

        subject_dirs = os.listdir(train_dir)
        print(f"找到 {len(subject_dirs)} 个主题目录")

        valid_subjects = 0
        for subject_dir in subject_dirs:
            subject_path = os.path.join(train_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            print(f"处理主题目录: {subject_path}")

            # 检查所有必要的文件是否存在 (使用.nii扩展名)
            required_files = {
                'seg': f'{subject_dir}_seg.nii',
                't1': f'{subject_dir}_t1.nii',
                't2': f'{subject_dir}_t2.nii',
                'flair': f'{subject_dir}_flair.nii',
                't1ce': f'{subject_dir}_t1ce.nii'
            }

            missing_files = []
            for file_type, file_name in required_files.items():
                if not os.path.exists(os.path.join(subject_path, file_name)):
                    missing_files.append(file_type)

            if missing_files:
                print(f"警告: 主题 {subject_dir} 缺少以下文件: {missing_files}")
                continue

            # 所有必要文件都存在，处理该主题
            valid_subjects += 1

            # 加载标签
            label_path = os.path.join(subject_path, required_files['seg'])
            try:
                label = nib.load(label_path).get_fdata()
            except Exception as e:
                print(f"无法加载标签文件 {label_path}: {e}")
                continue

            # 排除最低80层和最高26层
            valid_slices = 0
            for slice_idx in range(80, label.shape[2] - 26):
                slice_label = label[:, :, slice_idx]
                if np.sum(slice_label) == 0:
                    self.labels.append(0)  # 健康切片
                else:
                    self.labels.append(1)  # 病变切片

                # 加载四个通道的图像
                t1_path = os.path.join(subject_path, required_files['t1'])
                t2_path = os.path.join(subject_path, required_files['t2'])
                flair_path = os.path.join(subject_path, required_files['flair'])
                t1ce_path = os.path.join(subject_path, required_files['t1ce'])

                try:
                    t1 = nib.load(t1_path).get_fdata()[:, :, slice_idx]
                    t2 = nib.load(t2_path).get_fdata()[:, :, slice_idx]
                    flair = nib.load(flair_path).get_fdata()[:, :, slice_idx]
                    t1ce = nib.load(t1ce_path).get_fdata()[:, :, slice_idx]
                except Exception as e:
                    print(f"无法加载图像切片 {subject_dir} 索引 {slice_idx}: {e}")
                    # 如果加载某个模态失败，跳过当前切片
                    if self.labels:
                        self.labels.pop()  # 移除最后添加的标签
                    continue

                # 确保所有模态都有相同的尺寸
                if t1.shape != t2.shape or t1.shape != flair.shape or t1.shape != t1ce.shape:
                    print(f"警告: 主题 {subject_dir} 切片 {slice_idx} 的模态尺寸不一致")
                    if self.labels:
                        self.labels.pop()  # 移除最后添加的标签
                    continue

                image = np.stack([t1, t2, flair, t1ce], axis=0)
                self.image_paths.append(image)
                valid_slices += 1

            if valid_slices == 0:
                print(f"警告: 主题 {subject_dir} 没有有效切片")
                valid_subjects -= 1  # 不计入有效主题

        print(f"成功加载 {valid_subjects} 个主题，共 {len(self.image_paths)} 个样本")

        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 4, std=[0.5] * 4)  # 四个通道
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
        image = self.image_paths[idx]
        label = self.labels[idx]

        # 应用变换
        image = self.transform(image)

        # 训练时进行数据增强
        if self.augment and self.mode == 'train':
            image = self.augment_transform(image)

        target_label = 0  # 我们总是尝试生成健康图像

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'target_label': torch.tensor(target_label, dtype=torch.long)
        }


def get_dataloader(data_dir, image_size=256, batch_size=32, mode='train'):
    augment = (mode == 'train')
    dataset = BRATS2020Dataset(data_dir, image_size, mode, augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True
    )

    return dataloader