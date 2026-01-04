import os
import sys

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------
# Standard libs
# ------------------------------------------------------------------
import time
import argparse
import yaml
import torch
from attrdict import AttrDict

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
from orgmim.finetune.UNETR import (
    UNETR_base,
    UNETR_small,
    UNETR_large,
)

from orgmim.finetune.STUNet import (
    STUNet_base,
    STUNet_small,
    STUNet_large,
)

# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize segmentation model (CNN / ViT) with pretrained encoder"
    )
    parser.add_argument(
        "-c", "--cfg", type=str, default="orgmim",
        help="Configuration file name (without .yaml)"
    )
    parser.add_argument(
        "--arch", type=str,
        choices=["cnn", "vit"],
        default="vit",
        help="Backbone architecture"
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="CUDA visible devices"
    )
    return parser.parse_args()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_file = f"{args.cfg}.yaml"

    # --------------------------------------------------------------
    # Load config
    # --------------------------------------------------------------
    cfg_path = os.path.join("./config", cfg_file)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = AttrDict(yaml.safe_load(f))

    # bookkeeping (same style as pretraining)
    time_stamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    cfg.path = cfg_file
    cfg.time = time_stamp

    # --------------------------------------------------------------
    # Read model scale from YAML
    # --------------------------------------------------------------
    if not hasattr(cfg.MODEL, "scale"):
        raise KeyError("MODEL.scale must be specified in the config file")

    scale = cfg.MODEL.scale
    assert scale in ["small", "base", "large"], \
        f"Unsupported model scale: {scale}"

    # --------------------------------------------------------------
    # Print summary
    # --------------------------------------------------------------
    print("--------------------------------------------------")
    print(f"Config file   : {cfg_file}")
    print(f"Architecture  : {args.arch}")
    print(f"Model scale   : {scale}")
    print(f"Num classes   : {cfg.MODEL.num_classes}")
    print("--------------------------------------------------")

    # --------------------------------------------------------------
    # Build network
    # --------------------------------------------------------------
    if args.arch == "vit":
        # -------- UNETR --------
        if scale == "small":
            network = UNETR_small(cfg, pretrain=True)
        elif scale == "large":
            network = UNETR_large(cfg, pretrain=True)
        else:
            network = UNETR_base(cfg, pretrain=True)

    elif args.arch == "cnn":
        # -------- STUNet --------
        in_ch = 1
        out_ch = cfg.MODEL.num_classes

        if scale == "small":
            network = STUNet_small(in_ch, out_ch, pretrain=True)
        elif scale == "large":
            network = STUNet_large(in_ch, out_ch, pretrain=True)
        else:
            network = STUNet_base(in_ch, out_ch, pretrain=True)

    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # --------------------------------------------------------------
    # Move to device
    # --------------------------------------------------------------
    if torch.cuda.is_available():
        network = network.cuda()
        device = "cuda"
    else:
        device = "cpu"

    network.eval()

    print(f"[Init] Network constructed on {device}")
    print("[Init] Pretrained encoder loaded")
    print("*** Done ***")

# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
