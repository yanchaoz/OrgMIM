from collections import OrderedDict
import torch
def _load_stunet_pretrained_encoder_from_ckpt(model, checkpoint):
    pretrained_dict = checkpoint["model_weights"]

    new_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if "encoder" in k:
            new_k = k.split("sp_cnn.")[-1]
            new_dict[new_k] = v

    model.load_state_dict(new_dict, strict=False)
    print("[STUNet] Pretrained encoder loaded")


def _load_unetr_pretrained_encoder_from_ckpt(model, checkpoint):
    vit_state_dict = checkpoint["model_weights"]
    model.vit.load_state_dict(vit_state_dict, strict=False)
    print("[UNETR] Pretrained ViT encoder loaded")

def _download_pretrained_ckpt(url):
    """
    Download checkpoint automatically (cached by torch).
    """
    ckpt = torch.hub.load_state_dict_from_url(
        url,
        map_location="cpu",
        check_hash=False
    )
    return ckpt
