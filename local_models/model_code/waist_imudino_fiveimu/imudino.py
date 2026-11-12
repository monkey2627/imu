from __future__ import annotations

import math

import torch
import torch.nn as nn
try:
    from timm.layers.mlp import Mlp
    from timm.layers.norm import LayerNorm
    from timm.models.vision_transformer import Block
except ModuleNotFoundError:
    # Deployment-only evaluation operators for the existing PyTorch 2.1 runtime.
    from .torch_blocks import Mlp, LayerNorm, Block


def make_patch_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    mask: [B, L], where 1 = valid, 0 = invalid
    returns: [B, L // patch_size], where a patch is valid only if all tokens are valid
    """
    B, L = mask.shape
    if L % patch_size != 0:
        raise ValueError("L must be divisible by patch_size")

    patch_mask = mask.reshape(B, L // patch_size, patch_size).all(dim=-1)

    return patch_mask.to(mask.dtype)


class SinusoidalPE(nn.Module):
    """Frequency-aware positional encoding with a fixed 100 Hz reference."""

    def __init__(
        self,
        d_model: int,
        base: float = 100.0,
    ):
        super().__init__()

        self.d_model = int(d_model)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(base) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        # Keep Python exceptions out of the compiled graph.
        if not getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected x with shape [B, L, D], got {tuple(x.shape)}"
                )

            if x.shape[-1] != self.d_model:
                raise ValueError(
                    f"Expected feature dimension {self.d_model}, "
                    f"got {x.shape[-1]}"
                )

            if not isinstance(freqs, torch.Tensor):
                raise TypeError(
                    "freqs must be a Tensor when using torch.compile"
                )

            if freqs.ndim != 1 or freqs.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Expected freqs with shape [{x.shape[0]}], "
                    f"got {tuple(freqs.shape)}"
                )

            # This value-dependent check must only run outside compilation.
            if bool(torch.any(freqs <= 0)):
                raise ValueError("All sample frequencies must be positive.")

        B, L, D = x.shape

        # Pass freqs as a tensor on the same device whenever possible.
        freqs = freqs.to(device=x.device, dtype=x.dtype)

        positions = torch.arange(
            L,
            device=x.device,
            dtype=x.dtype,
        )

        div_term = self.div_term.to(
            device=x.device,
            dtype=x.dtype,
        )

        # [B, L, ceil(D / 2)]
        angles = (
            positions[None, :, None]
            * (100.0 / freqs[:, None, None])
            * div_term[None, None, :]
        )

        # Interleave sin and cos without in-place slice assignments:
        # [sin(a0), cos(a0), sin(a1), cos(a1), ...]
        pe = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(start_dim=-2)

        # Handles odd d_model values as well.
        pe = pe[..., :D]

        return x + pe


class IMUFormer(nn.Module):
    """IMU backbone with defaults matching the released checkpoint."""

    def __init__(
        self,
        patch_size=16,
        dim=768,
        num_heads=24,
        n_layers=12,
        n_reg_tokens=4,
        mlp_ratio=4,
        drop_path=0.3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.n_reg_tokens = n_reg_tokens
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path

        self.patchify = nn.Conv1d(6, dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02, a=-0.06, b=0.06)
        if self.n_reg_tokens > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, n_reg_tokens, dim))
            nn.init.trunc_normal_(self.reg_tokens, std=0.02, a=-0.06, b=0.06)
        self.register_buffer("special_token_masks", torch.ones(1, 1 + n_reg_tokens))

        self.positional_encoding = SinusoidalPE(dim)
        self.transformer = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=False,
                    act_layer=nn.GELU,
                    norm_layer=LayerNorm,
                    drop_path=drop_path,
                    mlp_layer=Mlp,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = LayerNorm(dim)

    def forward(
        self, input: dict[str, torch.Tensor | None]
    ) -> dict[str, torch.Tensor | None]:
        """Encode IMU samples.

        Input keys:
            imu_data: Float tensor [B, L, 6], accelerometer XYZ then gyroscope XYZ.
            freqs: Positive sampling frequencies in Hz, shape [B].
            masks: Optional [B, L] validity mask (1/True = valid).

        L must be a positive multiple of patch_size. Returns cls_token [B, D],
        reg_tokens [B, R, D] (None if R=0), and patch_tokens [B, L/patch_size, D].
        Padded patches remain in the output but are excluded as attention keys.
        """
        x = input["imu_data"]  # B, L, 6
        masks = input.get("masks")  # B, L or None
        freqs = input["freqs"]

        if x.ndim != 3 or x.shape[-1] != 6:
            raise ValueError("imu_data must have shape [B, L, 6]")
        B, L, _ = x.shape
        if L == 0 or L % self.patch_size != 0:
            raise ValueError("Sequence length must be a positive multiple of patch_size")
        if masks is not None and masks.shape != (B, L):
            raise ValueError(f"masks must have shape [{B}, {L}]")

        x = x.transpose(1, 2)
        patch_tokens = self.patchify(x).transpose(1, 2)

        if masks is not None:
            patch_masks = make_patch_mask(masks, self.patch_size)
        else:
            patch_masks = None

        patch_tokens = self.positional_encoding(patch_tokens, freqs)

        cls_token = self.cls_token.expand(B, -1, -1)
        if self.n_reg_tokens > 0:
            reg_tokens = self.reg_tokens.expand(B, -1, -1)
            all_tokens = torch.cat([cls_token, reg_tokens, patch_tokens], dim=1)
        else:
            all_tokens = torch.cat([cls_token, patch_tokens], dim=1)

        if masks is not None:
            all_token_masks = torch.cat(
                [self.special_token_masks.expand(B, -1), patch_masks], dim=1
            )
            attn_masks = all_token_masks[:, None, None, :].bool()
        else:
            attn_masks = None

        for layer in self.transformer:
            all_tokens = layer(all_tokens, attn_mask=attn_masks)
        all_tokens = self.final_norm(all_tokens)

        cls_token = all_tokens[:, 0, :]
        reg_tokens = (
            all_tokens[:, 1 : 1 + self.n_reg_tokens, :]
            if self.n_reg_tokens > 0 else None
        )
        patch_tokens = all_tokens[:, 1 + self.n_reg_tokens :, :]

        return {
            "cls_token": cls_token,
            "reg_tokens": reg_tokens,
            "patch_tokens": patch_tokens,
        }
