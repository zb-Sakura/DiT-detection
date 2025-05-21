import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random


def split_training_data(train_raw_dir, test_size=0.2, random_seed=42, stratified=True):
    """
    将训练数据划分为训练集和测试集

    Args:
        train_raw_dir: 原始训练数据目录
        test_size: 测试集比例
        random_seed: 随机种子
        stratified: 是否按标签分层划分
    Returns:
        train_subjects: 训练集样本ID列表
        test_subjects: 测试集样本ID列表
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 获取所有训练样本
    all_subjects = [d for d in os.listdir(train_raw_dir) if d.startswith("BraTS")]

    if not stratified:
        # 随机划分
        random.shuffle(all_subjects)
        split_idx = int(len(all_subjects) * (1 - test_size))
        train_subjects = all_subjects[:split_idx]
        test_subjects = all_subjects[split_idx:]
    else:
        # 分层划分（基于肿瘤存在性）
        subject_classes = []
        for subject in all_subjects:
            label_path = os.path.join(train_raw_dir, subject, f"{subject}_seg.nii")
            if os.path.exists(label_path):
                label_volume = nib.load(label_path).get_fdata()
                # 检查是否存在肿瘤（标签>0的体素）
                has_tumor = np.sum(label_volume > 0) > 0
                subject_classes.append(1 if has_tumor else 0)
            else:
                subject_classes.append(0)

        # 统计各类别的样本数
        from collections import Counter
        class_counts = Counter(subject_classes)
        print(f"类别分布: {class_counts}")

        # 检查是否有类别样本数少于2
        if any(count < 2 for count in class_counts.values()):
            print("警告: 某些类别样本数少于2，无法进行分层划分，将使用随机划分")
            random.shuffle(all_subjects)
            split_idx = int(len(all_subjects) * (1 - test_size))
            train_subjects = all_subjects[:split_idx]
            test_subjects = all_subjects[split_idx:]
        else:
            # 使用sklearn的train_test_split进行分层划分
            train_subjects, test_subjects = train_test_split(
                all_subjects, test_size=test_size, stratify=subject_classes, random_state=random_seed
            )

    # 打印划分结果
    print(f"训练集大小: {len(train_subjects)}")
    print(f"测试集大小: {len(test_subjects)}")

    return train_subjects, test_subjects


def load_brats_modalities(subject_dir, is_validation=False):
    """加载模态数据，validation模式下跳过标签"""
    modalities = ['t1', 't1ce', 't2', 'flair']
    volumes = []
    for mod in modalities:
        file_name = f"{os.path.basename(subject_dir)}_{mod}.nii"
        file_path = os.path.join(subject_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {mod} in {subject_dir}")
        volumes.append(nib.load(file_path).get_fdata())
    return np.stack(volumes, axis=-1)  # (H, W, D, C)


def preprocess_volume(volume, target_size=(256, 256), target_slice=128):
    """提取轴向切片、标准化、调整尺寸"""
    # 检查slice是否在有效范围内
    if target_slice >= volume.shape[2]:
        target_slice = volume.shape[2] // 2  # 如果指定slice超出范围，使用中间slice
        print(f"Warning: target_slice {target_slice} out of range. Using middle slice {target_slice}.")

    axial_slice = volume[:, :, target_slice, :]  # (H, W, C)
    normalized = np.zeros_like(axial_slice, dtype=np.float32)
    for c in range(axial_slice.shape[-1]):
        mask = axial_slice[..., c] != 0
        if np.any(mask):
            mean, std = np.mean(axial_slice[mask, c]), np.std(axial_slice[mask, c])
            normalized[..., c] = (axial_slice[..., c] - mean) / (std + 1e-5)
    resized = zoom(normalized, (target_size[0] / axial_slice.shape[0], target_size[1] / axial_slice.shape[1], 1),
                   order=1)
    return resized.transpose(2, 0, 1)  # (C, H, W)


def save_processed_data(input_dir, output_dir, subjects, is_validation=False, target_slice=128):
    """批量预处理指定的样本"""
    os.makedirs(output_dir, exist_ok=True)

    for subject in tqdm(subjects, desc=f"Processing {'validation' if is_validation else 'train'} data"):
        subject_path = os.path.join(input_dir, subject)
        try:
            volume = load_brats_modalities(subject_path, is_validation=is_validation)
            processed = preprocess_volume(volume, target_size=(256, 256), target_slice=target_slice)
            processed = torch.from_numpy(processed).float()  # 显式转换为张量
            torch.save(processed, os.path.join(output_dir, f"{subject}_processed.pt"))

            if not is_validation:
                label_path = os.path.join(subject_path, f"{os.path.basename(subject_path)}_seg.nii")
                if os.path.exists(label_path):
                    label_volume = nib.load(label_path).get_fdata()
                    # 确保标签slice与图像slice一致
                    label_slice = label_volume[:, :, min(target_slice, label_volume.shape[2] - 1)]
                    label_processed = zoom(label_slice, (256 / label_slice.shape[0], 256 / label_slice.shape[1]),
                                           order=0)
                    label_processed = (label_processed > 0).astype(np.float32)
                    torch.save(torch.from_numpy(label_processed), os.path.join(output_dir, f"{subject}_label.pt"))
        except Exception as e:
            print(f"Skipping {subject}: {e}")


def main():
    TARGET_SLICE = 128  # 定义要提取的轴向切片位置（默认中间位置）
    TEST_SIZE = 0.2  # 测试集比例
    RANDOM_SEED = 42  # 随机种子

    # 原始训练数据目录
    TRAIN_RAW_DIR = "./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    # 预处理后的数据目录
    PROCESSED_DIR = "./processed/brats2020"
    TRAIN_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "train")
    TEST_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "test")
    VAL_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "validation")

    # 创建主处理目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 划分训练集和测试集
    print("正在划分训练集和测试集...")
    train_subjects, test_subjects = split_training_data(
        train_raw_dir=TRAIN_RAW_DIR,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        stratified=True  # 启用分层划分，确保测试集的标签分布与训练集相似
    )

    # 保存划分结果
    split_info_path = os.path.join(PROCESSED_DIR, "train_test_split.txt")
    with open(split_info_path, 'w') as f:
        f.write(f"训练集样本 ({len(train_subjects)}):\n")
        f.write("\n".join(train_subjects) + "\n\n")
        f.write(f"测试集样本 ({len(test_subjects)}):\n")
        f.write("\n".join(test_subjects))

    print(f"训练/测试划分信息已保存至: {split_info_path}")

    # 预处理训练数据
    print("\n正在预处理训练数据...")
    save_processed_data(
        input_dir=TRAIN_RAW_DIR,
        output_dir=TRAIN_PROCESSED_DIR,
        subjects=train_subjects,
        is_validation=False,
        target_slice=TARGET_SLICE
    )

    # 预处理测试数据
    print("\n正在预处理测试数据...")
    save_processed_data(
        input_dir=TRAIN_RAW_DIR,
        output_dir=TEST_PROCESSED_DIR,
        subjects=test_subjects,
        is_validation=False,  # 测试集也需要处理标签
        target_slice=TARGET_SLICE
    )

    # 预处理Validation数据（保持原逻辑不变）
    print("\n正在预处理Validation数据...")
    VAL_RAW_DIR = "./data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
    val_subjects = [d for d in os.listdir(VAL_RAW_DIR) if d.startswith("BraTS")]
    save_processed_data(
        input_dir=VAL_RAW_DIR,
        output_dir=VAL_PROCESSED_DIR,
        subjects=val_subjects,
        is_validation=True,  # Validation数据不处理标签
        target_slice=TARGET_SLICE
    )

    print("\n数据预处理完成!")
    print(f"训练集保存至: {TRAIN_PROCESSED_DIR}")
    print(f"测试集保存至: {TEST_PROCESSED_DIR}")
    print(f"验证集保存至: {VAL_PROCESSED_DIR}")


if __name__ == "__main__":
    main()