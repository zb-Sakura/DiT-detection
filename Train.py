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
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
        dropout_prob=args.dropout_prob,
    )
    model.to(device)

    # 定义损失函数
    reconstruction_loss_fn = nn.MSELoss()
    anomaly_loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 创建数据加载器
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        mode='train'
    )

    val_loader = None
    if args.do_validation:
        try:
            val_loader = get_dataloader(
                data_dir=args.data_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                mode='validation'
            )
        except Exception as e:
            print(f"警告: 验证集加载失败 - {e}")
            args.do_validation = False  # 若加载失败，自动关闭验证

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
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

            # 计算重构损失
            recon_loss = reconstruction_loss_fn(x_healthy, images)

            # 创建异常目标
            anomaly_target = torch.zeros_like(anomaly_map)
            for j in range(images.shape[0]):
                if labels[j] == 1:  # 病变图像
                    anomaly_target[j] = 1.0

            # 计算异常损失
            anomaly_loss = anomaly_loss_fn(anomaly_map, anomaly_target)

            # 总损失
            loss = recon_loss + args.anomaly_weight * anomaly_loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}")

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, 平均训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                target_labels = batch['target_label'].to(device)
                t = torch.randint(0, args.num_timesteps, (images.shape[0],), device=device)

                x_healthy, anomaly_map = model(images, t, target_labels)
                recon_loss = reconstruction_loss_fn(x_healthy, images)

                anomaly_target = torch.zeros_like(anomaly_map)
                for j in range(images.shape[0]):
                    if labels[j] == 1:
                        anomaly_target[j] = 1.0

                anomaly_loss = anomaly_loss_fn(anomaly_map, anomaly_target)
                loss = recon_loss + args.anomaly_weight * anomaly_loss

                val_loss += loss.item()

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, 平均验证损失: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型到 {os.path.join(args.save_dir, 'best_model.pth')}")

        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch + 1}.pth'))
            print(f"保存模型到 {os.path.join(args.save_dir, f'model_epoch_{epoch + 1}.pth')}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    print(f"保存最终模型到 {os.path.join(args.save_dir, 'final_model.pth')}")


def test(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
        dropout_prob=args.dropout_prob,
    )
    model.to(device)

    # 加载最佳模型
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"加载最佳模型: {model_path}")
    else:
        print(f"未找到最佳模型: {model_path}")
        return

    # 定义损失函数
    reconstruction_loss_fn = nn.MSELoss()
    anomaly_loss_fn = nn.BCELoss()

    # 创建测试数据加载器
    test_loader = get_dataloader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        mode='validation'  # 假设验证集作为测试集
    )

    # 测试阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            target_labels = batch['target_label'].to(device)
            t = torch.randint(0, args.num_timesteps, (images.shape[0],), device=device)

            x_healthy, anomaly_map = model(images, t, target_labels)
            recon_loss = reconstruction_loss_fn(x_healthy, images)

            anomaly_target = torch.zeros_like(anomaly_map)
            for j in range(images.shape[0]):
                if labels[j] == 1:
                    anomaly_target[j] = 1.0

            anomaly_loss = anomaly_loss_fn(anomaly_map, anomaly_target)
            loss = recon_loss + args.anomaly_weight * anomaly_loss

            test_loss += loss.item()

    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader)
    print(f"平均测试损失: {avg_test_loss:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Medical Image Anomaly Detection with DiT')

    # 模式参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')

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

    # **新增参数**：是否在训练中启用验证
    parser.add_argument('--do_validation', action='store_true',
                        help='Enable validation during training (default: False)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


if __name__ == '__main__':
    main()