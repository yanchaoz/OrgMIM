import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from vit_pytorch.vit import Transformer
from typing import List, Optional, Tuple


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
    def generate_mask_mam(self, mam: torch.Tensor, step: int = 0, total_step: int = 400000, patch_size: int = 16, image_size: int = 128, alpha_t: float = 0.75) -> Tuple[
        torch.Tensor, torch.Tensor]:
        att_map = F.avg_pool3d(mam, kernel_size=patch_size, stride=patch_size)
        att_map = 1 - patchify_att(att_map, image_size, patch_size).squeeze()
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
    def generate_mask_alm(self, loss_pred: torch.Tensor, step: int = 0, total_step: int = 400000, alpha_t: float = 0.75) -> torch.Tensor:
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
    h, w, d = image_size//patch_size, image_size//patch_size, image_size//patch_size
    B, C = bchwd.shape[:2]
    bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
    bchwd = torch.einsum('bchpwqdg->bhwdpqgc', bchwd)
    bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3))
    return bln
