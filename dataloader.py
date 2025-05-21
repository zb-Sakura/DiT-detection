import torch
from torch.utils.data import Dataset, DataLoader
import os


class BRATS2020Dataset(Dataset):
    def __init__(self, data_dir, is_validation=False):
        self.data_dir = data_dir
        self.is_validation = is_validation  # 是否为validation数据集（无标签）
        self.subjects = [f for f in os.listdir(data_dir) if f.endswith("_processed.pt")]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        subject_id = subject.split("_processed")[0]

        # 加载图像 (C, H, W) = (4, 256, 256)
        img_path = os.path.join(self.data_dir, subject)
        image = torch.load(img_path)

        # Validation数据无标签，返回全零张量
        if self.is_validation:
            label = torch.zeros((1, 256, 256), dtype=torch.float32)
        else:
            # 加载标签 (1, H, W)
            label_path = os.path.join(self.data_dir, f"{subject_id}_label.pt")
            label = torch.load(label_path).unsqueeze(0) if os.path.exists(label_path) else torch.zeros((1, 256, 256))

        return {
            "image": image,
            "label": label,
            "subject_id": subject_id
        }


class BratsTestDataset(Dataset):
    """BraTS测试数据集 (带标签)"""

    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.samples = []

        # 获取所有样本ID
        for file in os.listdir(test_dir):
            if file.endswith("_processed.pt"):
                sample_id = file.split("_processed.pt")[0]
                self.samples.append(sample_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # 加载处理后的图像
        image_path = os.path.join(self.test_dir, f"{sample_id}_processed.pt")
        image = torch.load(image_path)

        # 加载标签
        label_path = os.path.join(self.test_dir, f"{sample_id}_label.pt")
        label = torch.load(label_path)

        return {"image": image, "label": label}

def get_brats_loaders(
        train_dir,
        val_dir,
        batch_size=4,
        num_workers=4
):
    """获取Train和Validation数据加载器"""
    train_dataset = BRATS2020Dataset(train_dir, is_validation=False)
    val_dataset = BRATS2020Dataset(val_dir, is_validation=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_brats_test_loader(test_dir, batch_size=1):
    """创建测试数据加载器"""
    dataset = BratsTestDataset(test_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不打乱顺序
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader
# 示例用法
if __name__ == "__main__":
    train_loader, val_loader = get_brats_loaders(
        data_dir="/path/to/BRATS2020/processed",
        batch_size=2,
        with_label=True
    )
    batch = next(iter(train_loader))
    print(f"Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")