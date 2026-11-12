from imudino import IMUFormer


def imuformer_huge_patch16() -> IMUFormer:
    """Build the released backbone; the historical name denotes a base-size model."""
    return IMUFormer(
        patch_size=16,
        dim=768,
        num_heads=24,
        n_layers=12,
        n_reg_tokens=4,
        mlp_ratio=4,
        drop_path=0.3,
    )
