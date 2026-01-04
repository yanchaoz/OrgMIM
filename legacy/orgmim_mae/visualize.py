import os
import sys
import random
from collections import OrderedDict
from pprint import pformat
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from matplotlib import pyplot as plt
import imageio

from vit_pytorch.vit import Transformer
from mim.vit_3d import ViT


class MAE(nn.Module):
    def __init__(
            self,
            *,
            encoder,
            decoder_dim,
            masking_ratio=0.75,
            decoder_depth=1,
            decoder_heads=8,
            decoder_dim_head=64,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.len_keep = round(num_patches * (1 - self.masking_ratio))
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, masked_indices, unmasked_indices, num_masked):
        device = img.device
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        masked_patches = patches[batch_range, masked_indices]
        encoded_tokens = self.encoder.transformer(tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        pred_pixel_values = torch.sigmoid(pred_pixel_values)
        return pred_pixel_values, masked_patches

    @torch.no_grad()
    def generate_mask_mam(self, mam: torch.Tensor, step: int = 0, total_step: int = 400000, patch_size: int = 16,
                          image_size: int = 128, alpha_t: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
        att_map = F.avg_pool3d(mam, kernel_size=patch_size, stride=patch_size)
        att_map = (1 - patchify_att(att_map, image_size, patch_size)).reshape(1, -1)
        B, L = att_map.shape
        ids_shuffle_att = torch.argsort(att_map, dim=1)
        ids_shuffle = torch.zeros_like(ids_shuffle_att, device=att_map.device).int()
        keep_ratio = float((step + 1) / total_step) * alpha_t

        if int((L - self.len_keep) * keep_ratio) <= 0:
            noise = torch.randn(B, L, device=att_map.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(B):
                len_loss = int((L - self.len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_att[i, -len_loss:]
                temp = torch.arange(L, device=att_map.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(att_map.device)

        unmasked_indices = ids_shuffle[:, :self.len_keep]
        masked_indices = ids_shuffle[:, self.len_keep:]
        return masked_indices, unmasked_indices

    @torch.no_grad()
    def generate_mask_alm(self, loss_pred: torch.Tensor, step: int = 0, total_step: int = 400000,
                          alpha_t: float = 0.75) -> torch.Tensor:
        B, L = loss_pred.shape
        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()
        keep_ratio = float((step + 1) / total_step) * alpha_t

        if int((L - self.len_keep) * keep_ratio) <= 0:
            noise = torch.randn(B, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(B):
                len_loss = int((L - self.len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]
                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

        unmasked_indices = ids_shuffle[:, :self.len_keep]
        masked_indices = ids_shuffle[:, self.len_keep:]
        return masked_indices, unmasked_indices


def patchify_att(bchwd, image_size, patch_size):
    p = 1
    h, w, d = image_size // patch_size, image_size // patch_size, image_size // patch_size
    B, C = bchwd.shape[:2]
    bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
    bchwd = torch.einsum('bchpwqdg->bhwdpqgc', bchwd)
    bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3))
    return bln



def reconstruct_and_visualize(
    learner,
    ckpt_paths,
    img,
    att,
    device,
    save_dir,
    name_list=None,
    mask_ratio=0.75,
    step=200000,
    total_step=400000,
    patch_size=16,
    image_size=128,
    alpha_t=1,
):

    os.makedirs(save_dir, exist_ok=True)
    num_masked = int(mask_ratio * 512)

    # === Generate dynamic mask ===
    masked_indices, unmasked_indices = learner.generate_mask_mam(
        att, step=step, total_step=total_step, patch_size=patch_size, image_size=image_size, alpha_t=alpha_t
    )

    for idx, ckpt_path in enumerate(ckpt_paths):
        name = name_list[idx] if name_list and idx < len(name_list) else f"model_{idx}"
        print(f"\n[INFO] Loading pre-trained model from {ckpt_path}")

        # --- Load checkpoint ---
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_weights', checkpoint)
        new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
        learner.load_state_dict(new_state_dict, strict=False)

        mae = learner.to(device)
        img = img.to(device)
        batch = img.shape[0]

        # --- Forward reconstruction ---
        with torch.no_grad():
            patches = mae.to_patch(img)
            batch, num_patches, *_ = patches.shape
            tokens = mae.patch_to_emb(patches)
            tokens = tokens + mae.encoder.pos_embedding[:, 1:(num_patches + 1)]
            batch_range = torch.arange(batch, device=device)[:, None]

            tokens = tokens[batch_range, unmasked_indices]
            masked_patches = patches[batch_range, masked_indices]

            encoded_tokens = mae.encoder.transformer(tokens)
            decoder_tokens = mae.enc_to_dec(encoded_tokens)
            unmasked_decoder_tokens = decoder_tokens + mae.decoder_pos_emb(unmasked_indices)

            mask_tokens = repeat(mae.mask_token, 'd -> b n d', b=batch, n=num_masked)
            mask_tokens = mask_tokens + mae.decoder_pos_emb(masked_indices)

            full_dec_tokens = torch.zeros(batch, num_patches, mae.decoder_dim, device=device)
            full_dec_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
            full_dec_tokens[batch_range, masked_indices] = mask_tokens

            decoded = mae.decoder(full_dec_tokens)
            mask_decoded = decoded[batch_range, masked_indices]
            pred_pixels = torch.sigmoid(mae.to_pixels(mask_decoded))

            # --- Reconstruction loss ---
            recon_loss = F.mse_loss(pred_pixels, masked_patches)
            print(f"[INFO] Reconstruction MSE Loss: {recon_loss.item():.6f}")

            # --- Recover full volume ---
            p1 = mae.encoder.image_patch_size
            p2 = mae.encoder.image_patch_size
            pf = mae.encoder.frame_patch_size
            h = mae.encoder.image_size // mae.encoder.image_patch_size
            w = h
            f = mae.encoder.frames // mae.encoder.frame_patch_size

            recons_tokens = torch.zeros(batch, num_patches, p1 * p2 * pf, device=device)
            mask_tokens = torch.zeros_like(recons_tokens)
            mask_tokens[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
            recons_tokens[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
            mask_tokens[batch_range, masked_indices] = 0
            recons_tokens[batch_range, masked_indices] = pred_pixels

            recons = rearrange(
                recons_tokens,
                'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',
                f=f, h=h, w=w, p1=p1, p2=p2, pf=pf, c=1
            )
            mask = rearrange(
                mask_tokens,
                'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',
                f=f, h=h, w=w, p1=p1, p2=p2, pf=pf, c=1
            )

        # --- Visualization ---
        x_c, y_c, z_c = 0, 0, 0
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(img[0, 0, z_c].cpu().numpy(), cmap='gray')
        plt.title('Raw')

        plt.subplot(1, 3, 2)
        plt.imshow(mask[0, 0, z_c].cpu().numpy(), cmap='gray')
        plt.title('Masked')

        plt.subplot(1, 3, 3)
        plt.imshow(recons[0, 0, z_c].cpu().numpy(), cmap='gray')
        plt.title('Reconstructed')

        save_path = os.path.join(save_dir, f'recons_{name}.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"[INFO] Saved reconstruction image to {save_path}")


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # =========================
    # 1. Environment setup
    # =========================
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # =========================
    # 2. Model initialization
    # =========================
    model = ViT(
        image_size=128,  # input image size
        frames=128,  # number of frames
        image_patch_size=16,  # image patch size
        frame_patch_size=16,  # frame patch size
        channels=1,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )

    learner = MAE(
        encoder=model,
        masking_ratio=0.75,
        decoder_dim=512,
        decoder_depth=6,
    )

    # =========================
    # 3. Paths & dataset
    # =========================
    ckpt_path_list = [
        '/opt/data/.../models/Dual_MAE/learner.ckpt',  # TODO
    ]
    img_path = '/opt/data/.../input/image.tif'  # TODO
    att_path = '/opt/data/.../input/att.tif'  # TODO
    save_dir = '/opt/data/.../output'  # TODO
    name_list = ['dual']

    # =========================
    # 4. Load 3D volumes
    # =========================
    img_tif = imageio.volread(img_path)
    mam_tif = imageio.volread(att_path)

    img = torch.tensor(img_tif)[0:128, 350:350 + 128, 150:150 + 128].reshape(1, 1, 128, 128, 128) / 255.0 # TODO
    mam = torch.tensor(mam_tif)[0:128, 140:140 + 128, 190:190 + 128].reshape(1, 1, 128, 128, 128) / 255.0 # TODO

    print(f"[INFO] Image shape: {tuple(img.shape)}, Attention shape: {tuple(att.shape)}")

    # =========================
    # 5. Reconstruction & Visualization
    # =========================
    reconstruct_and_visualize(
        learner=learner,
        ckpt_paths=ckpt_path_list,
        img=img,
        att=mam,
        device=device,
        save_dir=save_dir,
        name_list=name_list,
        mask_ratio=0.75,
        step=200000,
        total_step=400000,
        patch_size=16,
        image_size=128,
        alpha_t=1
    )