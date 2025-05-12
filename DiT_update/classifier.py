import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalAnomalyClassifier(nn.Module):
    """医学异常分类器（兼容DiT的潜在空间输入）"""

    def __init__(self, in_channels=4, num_classes=2, dim=64, time_emb_dim=256):
        super().__init__()

        # 时间步嵌入（与DiT同步）
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 4)
        )

        # 多尺度特征提取
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim * 2, 3, padding=1),
                nn.GroupNorm(16, dim * 2),
                nn.SiLU(),
                nn.Conv2d(dim * 2, dim * 2, 3, padding=1),
                nn.GroupNorm(16, dim * 2),
                nn.SiLU(),
                nn.MaxPool2d(2)
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim * 4, 3, padding=1),
                nn.GroupNorm(32, dim * 4),
                nn.SiLU(),
                nn.Conv2d(dim * 4, dim * 4, 3, padding=1),
                nn.GroupNorm(32, dim * 4),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        ])

        # 空间注意力
        self.attn = nn.Sequential(
            nn.Conv2d(dim * 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(dim * 4 + dim * 4, num_classes),  # 合并图像和时间特征
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: 潜在空间特征 [B, C, H, W]
            t: 时间步 [B,]
        Returns:
            logits: 分类logits [B, num_classes]
        """
        # 特征提取
        features = x
        for block in self.down_blocks[:-1]:
            features = block(features)

        # 注意力加权
        attn_map = self.attn(features)
        features = features * attn_map

        # 最终下采样
        features = self.down_blocks[-1](features)
        spatial_feat = features.flatten(1)  # [B, dim*4]

        # 时间嵌入
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # [B, dim*4]

        # 分类预测
        combined = torch.cat([spatial_feat, t_emb], dim=1)
        return self.head(combined)