import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import MedicalDiT  # 假设模型定义在此文件中
from dataloader import get_brats_test_loader  # 需要实现的测试数据加载器
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# 超参数配置
CONFIG = {
    "test_data_dir": r"./processed/brats2020/test",  # 带标签的测试集路径
    "model_path": "best_model.pth",  # 训练好的模型路径
    "batch_size": 1,  # 测试时通常用batch_size=1
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def calculate_dice(pred, target):
    """计算Dice系数 (预测值和目标值都应为二值化的)"""
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-5)
    return dice


def calculate_iou(pred, target):
    """计算IoU (Jaccard指数)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-5)
    return iou


def test_model():
    # 加载模型
    model = MedicalDiT().to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval()

    # 加载测试数据
    test_loader = get_brats_test_loader(
        test_dir=CONFIG["test_data_dir"],
        batch_size=CONFIG["batch_size"]
    )

    # 存储所有样本的评估结果
    all_dice_scores = []
    all_iou_scores = []
    all_auc_scores = []
    all_preds = []
    all_labels = []

    # 测试过程
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])

            # 模型推理
            t = torch.randint(0, 1000, (images.shape[0],), device=CONFIG["device"])
            _, anomaly_map = model(images, t, y=torch.zeros_like(t))

            # 将预测结果二值化 (阈值=0.5)
            preds_binary = (torch.sigmoid(anomaly_map) > 0.5).float()

            # 计算当前样本的指标
            for i in range(images.shape[0]):
                dice = calculate_dice(preds_binary[i].cpu().numpy(), labels[i].cpu().numpy())
                iou = calculate_iou(preds_binary[i].cpu().numpy(), labels[i].cpu().numpy())

                all_dice_scores.append(dice)
                all_iou_scores.append(iou)

                # 存储预测概率和真实标签用于计算AUC
                all_preds.extend(torch.sigmoid(anomaly_map[i]).flatten().cpu().numpy())
                all_labels.extend(labels[i].flatten().cpu().numpy())

    # 计算整体AUC
    if len(np.unique(all_labels)) > 1:  # 确保标签包含至少两个类别
        auc_score = roc_auc_score(all_labels, all_preds)
        # 计算PR曲线下面积
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        auprc = auc(recall, precision)
        all_auc_scores.append(auc_score)
        all_auc_scores.append(auprc)
    else:
        print("Warning: 测试集中只有一个类别，无法计算AUC。")
        auc_score = auprc = 0

    # 打印评估结果
    print(f"\n测试结果:")
    print(f"Dice系数: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"IoU: {np.mean(all_iou_scores):.4f} ± {np.std(all_iou_scores):.4f}")
    if len(all_auc_scores) > 0:
        print(f"AUC-ROC: {all_auc_scores[0]:.4f}")
        print(f"AUC-PR: {all_auc_scores[1]:.4f}")

    return {
        "dice": np.mean(all_dice_scores),
        "iou": np.mean(all_iou_scores),
        "auc_roc": all_auc_scores[0] if len(all_auc_scores) > 0 else 0,
        "auc_pr": all_auc_scores[1] if len(all_auc_scores) > 0 else 0,
    }


if __name__ == "__main__":
    metrics = test_model()