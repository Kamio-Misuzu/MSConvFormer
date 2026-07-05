import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionPool(nn.Module):
    """
    软注意力池化时间维度（替换 max/mean pool），初始化为均匀权重更稳定。
    输入:  (B, T, D)
    输出:  (B, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.attn.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        score = self.attn(torch.tanh(x)).squeeze(-1)  # (B, T)
        if key_padding_mask is not None:
            # key_padding_mask: True 表示 padding（不可用位置）
            score = score.masked_fill(key_padding_mask, float("-inf"))
        alpha = torch.softmax(score, dim=-1)  # (B, T)
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        if return_alpha:
            return pooled, alpha
        return pooled


class MaskedMeanPool(nn.Module):
    """Masked mean pooling over time for (B, T, D)."""

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if key_padding_mask is None:
            return x.mean(dim=1)
        valid = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)  # (B, T, 1); 1=valid, 0=pad
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom


def manifold_mixup(z: torch.Tensor, alpha: float = 0.4):
    """
    z: (B, D)
    返回: (z_mix, lam, idx) 供训练循环计算 mixup loss
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(z.size(0), device=z.device)
    z_mix = lam * z + (1.0 - lam) * z[idx]
    return z_mix, lam, idx


class DWConv1d(nn.Module):
    """Depthwise + Pointwise 1D Conv."""

    def __init__(self, dim: int, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation, groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = F.gelu(x)
        return self.dropout(x)


class MultiScaleTemporalBlock(nn.Module):
    """
    多尺度时域卷积：并行不同 kernel/dilation，适配 ECG(形态) 与 EEG(频段/节律)。
    输入/输出: (B, D, T)
    """

    def __init__(
        self,
        dim: int,
        kernels: tuple[int, int, int] = (3, 7, 15),
        dilations: tuple[int, int, int] = (1, 2, 4),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(kernels) == len(dilations) == 3
        self.branches = nn.ModuleList(
            [DWConv1d(dim, k, d, dropout=dropout) for k, d in zip(kernels, dilations)]
        )
        self.fuse = nn.Conv1d(dim * 3, dim, 1, bias=False)
        self.fuse_norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        ys = [b(x) for b in self.branches]
        y = torch.cat(ys, dim=1)  # (B, 3D, T)
        y = self.fuse(y)
        y = self.fuse_norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return x + y


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention


        self.use_mixup = getattr(configs, "use_mixup", True)
        self.mixup_alpha = float(getattr(configs, "mixup_alpha", 0.4))
        self.mixup_prob = float(getattr(configs, "mixup_prob", 0.5))

        # Ablation switches
        self.no_ms_blocks = bool(getattr(configs, "no_ms_blocks", False))
        self.no_transformer = bool(getattr(configs, "no_transformer", False))
        self.pool_type = str(getattr(configs, "pool", "attn"))

        enc_in = int(configs.enc_in)
        d_model = int(getattr(configs, "d_model", 128))
        e_layers = int(getattr(configs, "e_layers", 4))
        dropout = float(getattr(configs, "dropout", 0.1))
        n_heads = int(getattr(configs, "n_heads", 8))
        d_ff = int(getattr(configs, "d_ff", d_model * 4))


        # Stem: (B, T, C) -> (B, T, D)
        self.stem = nn.Sequential(
            nn.Linear(enc_in, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-scale temporal conv (token-mixing): operate on (B, D, T)
        if self.no_ms_blocks:
            self.ms_blocks = nn.Identity()
        else:
            self.ms_blocks = nn.Sequential(
                *[MultiScaleTemporalBlock(d_model, dropout=dropout) for _ in range(max(1, e_layers // 2))]
            )

        # Lightweight transformer encoder (global dependencies)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, n_heads),
            dim_feedforward=max(d_model, d_ff),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if self.no_transformer:
            self.transformer = nn.Identity()
        else:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, e_layers - (e_layers // 2)))

        self.pool = AttentionPool(d_model) if self.pool_type == "attn" else MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, int(configs.num_class)),
        )



    def classification(self, x_enc: torch.Tensor, padding_mask: torch.Tensor | None):


        x = self.stem(x_enc)  # (B, T, D)

        # conv blocks need (B, D, T)
        x_conv = self.ms_blocks(x.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        x = x + x_conv

        key_padding_mask = None
        if padding_mask is not None:
            # padding_mask: 1/True = keep, transformer expects True = pad
            key_padding_mask = ~padding_mask.bool()

        if isinstance(self.transformer, nn.Identity):
            x = x
        else:
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, T, D)
        z = self.pool(x, key_padding_mask=key_padding_mask)  # (B, D)

        if self.training and self.use_mixup and torch.rand(1, device=z.device).item() < self.mixup_prob:
            z_mix, lam, idx = manifold_mixup(z, alpha=self.mixup_alpha)
            logits = self.head(z_mix)
            return logits, lam, idx

        logits = self.head(z)
        return logits

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None




