from einops import rearrange,repeat
import torch
import torch.nn.functional as F
from skimage.feature import hog
from matplotlib import pyplot as plt
import random
import os
import numpy as np
from typing import List, Optional, Tuple

def visual_2d(mae, img, att, save_dir, iters, num = 5):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    with torch.no_grad():
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # get patches
        mae = mae.module if hasattr(mae, 'module') else mae
        mae = mae.to(device)
        img = img.to(device)
        patches = mae.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = mae.patch_to_emb(patches)
        tokens = tokens + mae.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(mae.masking_ratio * num_patches)
        masked_indices, unmasked_indices = generate_mask_mam(att, keep_ratio = 1/2)

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        encoded_tokens = mae.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = mae.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + mae.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(mae.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + mae.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, mae.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = mae.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = mae.to_pixels(mask_tokens)
        pred_pixel_values = torch.sigmoid(pred_pixel_values)
                
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        # get the reconstructed image
        p1 = mae.encoder.image_patch_size
        p2 = mae.encoder.image_patch_size
        pf = mae.encoder.frame_patch_size
        h = mae.encoder.image_size // mae.encoder.image_patch_size
        w = mae.encoder.image_size // mae.encoder.image_patch_size
        f = mae.encoder.frames // mae.encoder.frame_patch_size
        recons_tokens = torch.zeros(batch, num_patches, p1 * p2 * pf, device=device)
        mask_tokens = torch.zeros(batch, num_patches, p1 * p2 * pf, device=device)
        mask_tokens[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
        recons_tokens[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
        mask_tokens[batch_range, masked_indices] = torch.zeros_like(mask_tokens[batch_range, masked_indices])
        recons_tokens[batch_range, masked_indices] = pred_pixel_values

        recons = rearrange(recons_tokens, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',f = f, h = h, w=w, p1 = p1, \
                             p2 = p2, pf = pf, c = 1)
        mask = rearrange(mask_tokens, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)',f = f, h = h, w=w, p1 = p1, \
                                p2 = p2, pf = pf, c = 1)
        
        for i in range(num):
            x, y, z, _, _ = img.shape
            x_c, y_c, z_c = random.randint(0,x-1), random.randint(0,y-1), random.randint(0,z-1)
            plt.figure(figsize=(10,10))
            plt.subplot(1,3,1)
            plt.imshow(img[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.title('raw')
        
            plt.subplot(1,3,2)
            plt.imshow(mask[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.xlabel('reconstruction loss: {}'.format(recon_loss.item()))
            plt.title('masked')
            
            plt.subplot(1,3,3)
            plt.imshow(recons[x_c, y_c, z_c].cpu().numpy(),cmap='gray')
            plt.title('reconstructed')

            plt.savefig(os.path.join(save_dir, 'recons_{}_{}.png'.format(iters, i)), dpi=400, bbox_inches='tight')

    torch.cuda.empty_cache()


@torch.no_grad()
def generate_mask_mam(mam: torch.Tensor, patch_size: int = 16, image_size: int = 128, keep_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
    len_keep = round((image_size/patch_size)**3 * 0.25)
    att_map = F.avg_pool3d(mam, kernel_size=patch_size, stride=patch_size)
    att_map = 1 - patchify_att(att_map, image_size, patch_size).squeeze()
    B, L = att_map.shape
    ids_shuffle_att = torch.argsort(att_map, dim=1)
    ids_shuffle = torch.zeros_like(ids_shuffle_att, device=att_map.device).int()

    if int((L - len_keep) * keep_ratio) <= 0:
        noise = torch.randn(B, L, device=att_map.device)
        ids_shuffle = torch.argsort(noise, dim=1)
    else:
        for i in range(B):
            len_loss = int((L - len_keep) * keep_ratio)
            ids_shuffle[i, -len_loss:] = ids_shuffle_att[i, -len_loss:]
            temp = torch.arange(L, device=att_map.device)
            deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
            np.random.shuffle(deleted)
            ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(att_map.device)

    unmasked_indices = ids_shuffle[:, :len_keep]
    masked_indices = ids_shuffle[:, len_keep:]
    return masked_indices, unmasked_indices


def patchify_att(bchwd, image_size, patch_size):
    p = 1
    h, w, d = image_size//patch_size, image_size//patch_size, image_size//patch_size
    B, C = bchwd.shape[:2]
    bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
    bchwd = torch.einsum('bchpwqdg->bhwdpqgc', bchwd)
    bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3)) 
    return bln
