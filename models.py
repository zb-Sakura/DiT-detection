import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import math


# DiT模型核心组件
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_output

        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels=4, input_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_size = input_size  # 输入图像尺寸，用于最终调整大小
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)  # 输出维度：16*16*4=1024
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        # x: (batch, num_patches, hidden_size) = (2, 256, 768)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # c: (2, 768)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (2, 256, 768)

        # 线性层输出：(batch, num_patches, patch_size^2*out_channels) = (2, 256, 1024)
        x = self.linear(x)

        # 调整维度为 (batch, patch_size^2*out_channels, num_patches)
        x = x.permute(0, 2, 1)  # (2, 1024, 256)

        # 重构为 (batch, out_channels, patch_size, num_patches^(1/2))
        num_patches_per_side = int(self.input_size // self.patch_size)  # 16
        x = x.reshape(-1, self.out_channels, self.patch_size, num_patches_per_side)  # (2, 4, 16, 16)

        # 拼接所有Patch为完整图像（16x16个16x16的Patch → 256x256）
        x = x.permute(0, 1, 3, 2).reshape(-1, self.out_channels, self.input_size, self.input_size)  # (2, 4, 256, 256)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        # 正确计算：(input_size // patch_size) ** 2
        self.num_patches = (input_size // patch_size) ** 2  # 应为 256（256/16=16; 16^2=256）
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        x = self.proj(x)  # (batch, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)
        x = x.transpose(1,2)# (batch, num_patches, embed_dim)
        return x  # 维度正确：(batch, num_patches, hidden_size)


class DiT(nn.Module):
    def __init__(
            self,
            input_size=256,
            patch_size=16,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=2,
            dropout_prob=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_channels)

        # 调用权重初始化方法
        self.initialize_weights()

    def initialize_weights(self):
        """安全初始化权重，处理LayerNorm无参数的情况"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # 仅当存在weight/bias时初始化（如elementwise_affine=True）
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, y):
        # 图像嵌入
        x = self.x_embedder(x)
        x = x + self.pos_embed
        # 时间步和类别嵌入
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        # Transformer块处理
        for block in self.blocks:
            x = block(x, c)
        # 最终层重构
        x = self.final_layer(x, c)
        return x


# 医学图像异常检测专用的DiT模型
class MedicalDiT(nn.Module):
    def __init__(
            self,
            input_size=256,  # 修改为256
            patch_size=16,
            in_channels=4,  # 修改为4
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=2,  # 健康/病变
            dropout_prob=0.1,
    ):
        super().__init__()
        self.dit = DiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )

        # 用于异常检测的编码器（病变到健康的映射）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 用于生成异常图的差异模块
        self.diff_module = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t, y):
        # 使用DiT生成健康图像
        x_healthy = self.dit(x, t, y)
        print(f"x shape: {x.shape}")  # 应输出 torch.Size([batch, 4, 256, 256])
        print(f"x_healthy shape: {x_healthy.shape}")  # 应输出 torch.Size([batch, 4, 256, 256])
        # 生成异常图
        concat = torch.cat([x, x_healthy], dim=1)
        anomaly_map = self.diff_module(concat)

        return x_healthy, anomaly_map