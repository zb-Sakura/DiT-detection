import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
from models import MedicalDiT
from dataloader import get_dataloader
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(args):
    # 创建保存模型的目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = MedicalDiT(
        input_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        dropout_prob=args.dropout_prob
    ).to(device)

    # 如果使用多GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 定义损失函数
    reconstruction_loss_fn = nn.MSELoss()
    anomaly_loss_fn = nn.BCELoss()

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 获取数据加载器
    train_loader = get_dataloader(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        mode='train'
    )

    val_loader = get_dataloader(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        mode='val'
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_anomaly_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            # 获取数据并移至设备
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            target_labels = batch['target_label'].to(device)

            # 随机采样时间步
            t = torch.randint(0, args.num_timesteps, (images.shape[0],), device=device)

            # 前向传播
            optimizer.zero_grad()
            x_healthy, anomaly_map = model(images, t, target_labels)

            # 计算损失
            recon_loss = reconstruction_loss_fn(x_healthy, images)

            # 为病变图像创建二值异常图（1表示异常，0表示正常）
            anomaly_target = torch.zeros_like(anomaly_map)
            for j in range(images.shape[0]):
                if labels[j] == 1:  # 病变图像
                    anomaly_target[j] = 1.0

            anomaly_loss = anomaly_loss_fn(anomaly_map, anomaly_target)

            # 总损失
            loss = recon_loss + args.anomaly_weight * anomaly_loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 记录损失
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_anomaly_loss += anomaly_loss.item()

            # 更新进度条
            progress_bar.set_description(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Loss: {loss.item():.4f} "
                f"Recon: {recon_loss.item():.4f} "
                f"Anomaly: {anomaly_loss.item():.4f}"
            )

        # 计算平均损失
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_anomaly_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                target_labels = batch['target_label'].to(device)

                # 随机采样时间步
                t = torch.randint(0, args.num_timesteps, (images.shape[0],), device=device)

                # 前向传播
                x_healthy, anomaly_map = model(images, t, target_labels)

                # 计算损失
                recon_loss = reconstruction_loss_fn(x_healthy, images)

                # 为病变图像创建二值异常图
                anomaly_target = torch.zeros_like(anomaly_map)
                for j in range(images.shape[0]):
                    if labels[j] == 1:  # 病变图像
                        anomaly_target[j] = 1.0

                anomaly_loss = anomaly_loss_fn(anomaly_map, anomaly_target)

                # 总损失
                loss = recon_loss + args.anomaly_weight * anomaly_loss

                val_loss += loss.item()

        # 计算平均验证损失
        val_loss /= len(val_loader)

        # 学习率调整
        scheduler.step()

        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Recon Loss: {train_recon_loss:.4f} "
              f"Anomaly Loss: {train_anomaly_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f'model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"Saved best model at epoch {epoch + 1} with val loss: {val_loss:.4f}")

        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"Saved model at epoch {epoch + 1}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Medical Image Anomaly Detection with DiT')

    # 模型参数
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for DiT')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size for DiT')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (healthy/lesion)')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='Dropout probability')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--anomaly_weight', type=float, default=1.0, help='Weight for anomaly loss')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')

    # 数据和保存参数
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every n epochs')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()