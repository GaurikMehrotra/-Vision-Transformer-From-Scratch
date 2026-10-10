"""Report parameters and fvcore-counted FLOPs; timm is a baseline reference only."""

import argparse
from functools import partial
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from .utils import build_model, load_config, save_json


def profile_model(model: torch.nn.Module, images: torch.Tensor) -> dict:
    """Count recognized operations (1 multiply-add = 1 FLOP), disclose omissions."""
    analysis = FlopCountAnalysis(model.eval(), images)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = analysis.total()
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "counted_flops": total,
        "unsupported_operators": dict(analysis.unsupported_ops()),
        "convention": "one multiply-add = one FLOP; omitted operators listed explicitly",
    }


def timm_reference(config: dict) -> torch.nn.Module:
    """Equivalent learned-position or no-position timm baseline; never model internals."""
    from timm.models.vision_transformer import VisionTransformer

    m = config["model"]
    if m["name"] != "vit" or m["positional_encoding"] not in {"learned", "none"}:
        raise ValueError("timm reference supports ViT learned/none positional configurations")
    if m["patch_method"] != "conv":
        raise ValueError("Use conv patch embedding for timm parameter correspondence")
    model = VisionTransformer(
        img_size=m["image_size"],
        patch_size=m["patch_size"],
        in_chans=m["in_channels"],
        num_classes=m["num_classes"],
        embed_dim=m["embed_dim"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        global_pool="token" if m["pooling"] == "cls" else "avg",
        class_token=m["pooling"] == "cls",
        fc_norm=False,
        pos_embed="learn",
        norm_layer=partial(torch.nn.LayerNorm, eps=m["norm_eps"]),
    )
    if m["positional_encoding"] == "none":
        # timm 1.0.15 skips CLS insertion when pos_embed=None. A zero buffer
        # preserves CLS semantics without adding learned position parameters.
        zeros = torch.zeros_like(model.pos_embed)
        del model.pos_embed
        model.register_buffer("pos_embed", zeros)
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--compare-timm", action="store_true")
    parser.add_argument("--output", default="reports/profile.json")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["training"]["cpu_threads"])
    images = torch.randn(1, 3, config["data"]["image_size"], config["data"]["image_size"])
    result = {"ours": profile_model(build_model(config), images)}
    if args.compare_timm:
        result["timm"] = profile_model(timm_reference(config), images)
        result["parameters_match"] = result["ours"]["parameters"] == result["timm"]["parameters"]
        result["counted_flops_match"] = (
            result["ours"]["counted_flops"] == result["timm"]["counted_flops"]
        )
    save_json(result, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
