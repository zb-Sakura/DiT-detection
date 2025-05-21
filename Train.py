import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MedicalDiT  # 假设你的模型定义在models.py中
from dataloader import get_brats_loaders
import wandb  # 可选，用于日志记录

# 超参数配置
CONFIG = {
    # 数据路径 - 替换为你实际的预处理数据路径
    "train_data_dir": r"./processed/brats2020/train",
    "val_data_dir": r"./processed/brats2020/validation",

    # 训练参数
    "batch_size": 2,
    "epochs": 50,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 模型参数（确保与MedicalDiT定义一致）
    "input_size": 256,
    "patch_size": 16,
    "in_channels": 4,
    "hidden_size": 768,
    "depth": 12,
    "num_heads": 12,
}


def train():
    wandb.init(project="brats2020-medical-dit", config=CONFIG)
    model = MedicalDiT().to(CONFIG["device"])
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # 加载数据（Validation无标签，训练时不计算anomaly_loss）
    train_loader, val_loader = get_brats_loaders(
        train_dir=CONFIG["train_data_dir"],
        val_dir=CONFIG["val_data_dir"],
        batch_size=CONFIG["batch_size"]
    )

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        # Train阶段（有标签，计算双损失）
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")
        for batch in progress_bar:
            images = batch["image"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])
            t = torch.randint(0, 1000, (images.shape[0],), device=CONFIG["device"])

            x_healthy, anomaly_map = model(images, t, y=torch.zeros_like(t))
            recon_loss = mse_loss(x_healthy, images)
            anomaly_loss = nn.BCEWithLogitsLoss()(anomaly_map, labels)
            loss = recon_loss + anomaly_loss  # 假设anomaly_weight=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            wandb.log({"train_loss": loss.item()})

        # Validation阶段（无标签，仅计算重建损失）
        model.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]")
            for batch in progress_bar:
                images = batch["image"].to(CONFIG["device"])
                t = torch.randint(0, 1000, (images.shape[0],), device=CONFIG["device"])

                x_healthy, _ = model(images, t, y=torch.zeros_like(t))
                val_recon_loss += mse_loss(x_healthy, images).item()

        val_recon_loss /= len(val_loader)
        scheduler.step()
        wandb.log({"val_recon_loss": val_recon_loss})

        # 保存模型（基于验证重建损失）
        if val_recon_loss < wandb.run.summary.get("best_val_loss", float("inf")):
            torch.save(model.state_dict(), "best_model.pth")
            wandb.run.summary["best_val_loss"] = val_recon_loss

        print(
            f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Recon Loss: {val_recon_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    train()