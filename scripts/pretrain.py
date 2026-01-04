import sys
import os
import time
import argparse
import yaml
from attrdict import AttrDict

from orgmim.trainers.init_project import init_project, load_dataset


def get_loop_fn(model_type: str):
    if model_type == "vit":
        from orgmim.trainers.pretrain_vit import loop
    elif model_type == "cnn":
        from orgmim.trainers.pretrain_cnn import loop
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return loop


def parse_args():
    parser = argparse.ArgumentParser(description="OrgMIM pretraining")
    parser.add_argument(
        "-c", "--cfg", type=str, default="orgmim",
        help="Configuration file name (without .yaml)"
    )

    parser.add_argument(
        "--gpu", type=str, default="3",
        help="CUDA visible devices"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_file = f"{args.cfg}.yaml"

    with open(os.path.join("./config", cfg_file), "r") as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    time_stamp = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    cfg.path = cfg_file
    cfg.time = time_stamp

    writer = init_project(cfg)
    train_provider = load_dataset(cfg)

    init_iters = 0
    loop_fn = get_loop_fn(cfg.MODEL.arch)
    loop_fn(cfg, train_provider, init_iters, writer)

    writer.close()
    print("*** Done ***")


if __name__ == "__main__":
    main()
