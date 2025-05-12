import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import argparse
import logging
from datetime import datetime
from medical_classifier import MedicalAnomalyClassifier
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


class MedicalTrainer:
    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.local_rank}'
        self._setup_distributed()
        self._init_models()
        self._create_optimizers()
        self.scaler = GradScaler(enabled=args.amp)

    def _setup_distributed(self):
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank)
        if self.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)
            self._init_logging()

    def _init_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.args.output_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_models(self):
        # 初始化VAE（用于潜在空间编码）
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.args.vae}").to(self.device)
        self.vae.requires_grad_(False)

        # 初始化分类器
        self.classifier = MedicalAnomalyClassifier(
            in_channels=4,  # 匹配VAE输出通道
            num_classes=1 if self.args.binary else 2,
            dim=self.args.dim
        ).to(self.device)
        self.classifier = DDP(self.classifier, device_ids=[self.args.local_rank])

        # 初始化扩散模型（用于噪声生成）
        self.diffusion = create_diffusion(timestep_respacing="")

    def _create_optimizers(self):
        self.opt = AdamW(
            self.classifier.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )

    def train_epoch(self, dataloader, epoch):
        self.classifier.train()
        total_loss = 0.0

        for step, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.float().to(self.device) if self.args.binary else labels.long().to(self.device)

            # VAE编码到潜在空间
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample() * 0.18215

            # 随机时间步和噪声
            t = torch.randint(0, self.diffusion.num_timesteps, (images.size(0),), device=self.device)
            noisy_latents = self.diffusion.q_sample(latents, t)

            # 混合精度训练
            with autocast(enabled=self.args.amp):
                logits = self.classifier(noisy_latents, t)
                if self.args.binary:
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)
                else:
                    loss = F.cross_entropy(logits, labels)

            # 反向传播
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            total_loss += loss.item()

            # 日志记录
            if step % self.args.log_steps == 0 and self.is_main_process:
                self.logger.info(
                    f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | LR: {self.opt.param_groups[0]['lr']:.2e}"
                )

        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch):
        if not self.is_main_process:
            return

        state = {
            'epoch': epoch,
            'model': self.classifier.module.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'args': self.args
        }
        torch.save(state, os.path.join(self.args.output_dir, f'checkpoint_epoch{epoch}.pt'))
        self.logger.info(f"Saved checkpoint at epoch {epoch}")

    @property
    def is_main_process(self):
        return self.args.local_rank == 0


def main():
    parser = argparse.ArgumentParser(description='Medical Classifier Training')

    # 训练参数
    parser.add_argument('--data-dir', type=str, required=True, help='Path to medical dataset')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--dim', type=int, default=64, help='Base channel dimension')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--log-steps', type=int, default=50, help='Logging frequency')

    # 医学数据参数
    parser.add_argument('--binary', action='store_true', help='Binary classification mode')
    parser.add_argument('--vae', choices=['ema', 'mse'], default='ema', help='VAE type')

    # 分布式参数
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')

    args = parser.parse_args()

    trainer = MedicalTrainer(args)

    # 数据加载器（需替换为您的医学数据集）
    dataset = YourMedicalDataset(args.data_dir)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 训练循环
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        avg_loss = trainer.train_epoch(dataloader, epoch)

        if trainer.is_main_process:
            trainer.logger.info(f"Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f}")
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                trainer.save_checkpoint(epoch)


if __name__ == '__main__':
    main()