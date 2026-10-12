"""MARA model: motion adaptation, frozen MOMENT encoder, and classifier."""

import torch
from torch import nn
from torch.nn import functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


def _diff(x: torch.Tensor) -> torch.Tensor:
    return x - torch.cat((x[:, :1], x[:, :-1]), dim=1)


class MARAAdapter(nn.Module):
    """Map [B, T, C] sensor windows to motion-enhanced windows."""

    def __init__(
        self,
        n_channels: int,
        a1: float = 0.2,
        a2: float = 0.1,
        kernel_size: int = 3,
        smooth_kernel: int = 7,
        scale: float = 0.065,
        jerk_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if n_channels < 1 or any(k < 1 or k % 2 == 0 for k in (kernel_size, smooth_kernel)):
            raise ValueError("Channels must be positive and kernels must be positive odd integers.")
        if not (0 <= a1 <= 1 and 0 <= a2 <= 1):
            raise ValueError("a1 and a2 must each lie in [0, 1].")
        self.n_channels = n_channels
        self.smooth_kernel = smooth_kernel
        for name, value in (("a1", a1), ("a2", a2), ("scale", scale)):
            self.register_buffer(name, torch.tensor(float(value)))
        self.diff1_conv = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=kernel_size // 2, groups=n_channels, bias=True,
        )
        self.diff2_conv = nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=kernel_size // 2, groups=n_channels, bias=True,
        )
        self.branch_gain = nn.Parameter(torch.ones(n_channels))
        self.jerk_gate = nn.Parameter(torch.full((n_channels,), float(jerk_gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.n_channels or x.shape[1] == 0:
            raise ValueError(f"Expected [B, T, {self.n_channels}] with T > 0.")
        kernel = min(self.smooth_kernel, x.shape[1])
        kernel -= 1 - kernel % 2
        smoothed = F.avg_pool1d(
            F.pad(x.transpose(1, 2), (kernel // 2, kernel // 2), mode="replicate"),
            kernel, stride=1,
        ).transpose(1, 2)
        d1_raw = _diff(smoothed)
        d2_raw = _diff(d1_raw)
        d1, d2 = _normalize(d1_raw), _normalize(d2_raw)
        y1 = self.diff1_conv(d1.transpose(1, 2)).transpose(1, 2)
        y2 = self.diff2_conv(d2.transpose(1, 2)).transpose(1, 2)
        jerk = _normalize(d2 / d1.abs().clamp_min(1e-3))
        delta = self.branch_gain * (self.a1 * y1 + self.a2 * y2) + self.jerk_gate * jerk
        return x + self.scale * delta


class MARA(nn.Module):
    """Wrap an initialized MOMENT classification model using concat reduction.

    Input: [B, T, C]. Output: [B, num_classes] logits.
    """

    def __init__(self, backbone: nn.Module, n_channels: int, **adapter_kwargs) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.head.requires_grad_(True)
        self.adapter = MARAAdapter(n_channels, **adapter_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from momentfm.utils.masking import Masking

        model = self.backbone
        x = self.adapter(x).transpose(1, 2)
        batch, channels, length = x.shape
        mask = x.new_ones((batch, length))
        x = model.normalizer(x=x, mask=mask, mode="norm")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = model.patch_embedding(model.tokenizer(x=x), mask=mask)
        patches, dim = tokens.shape[-2:]
        patch_mask = Masking.convert_seq_to_patch_view(mask, model.patch_len)
        hidden = model.encoder(
            inputs_embeds=tokens.reshape(batch * channels, patches, dim),
            attention_mask=patch_mask.repeat_interleave(channels, dim=0),
        ).last_hidden_state
        hidden = hidden.reshape(batch, channels, patches, dim)
        features = hidden.permute(0, 2, 3, 1).reshape(batch, patches, dim * channels).mean(dim=1)
        return model.head.linear(model.head.dropout(features))
