from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from skimage.feature import hog
from matplotlib import pyplot as plt
import random
import os


def visual_2d(mae, img, save_dir, iters):
    """
    Visualize the raw, masked, and reconstructed images in 2D.

    Args:
        mae: The model used for reconstruction.
        img: Input image tensor.
        save_dir: Directory to save the visualization.
        iters: Current iteration number for naming the saved files.
    """
    with torch.no_grad():
        # Get raw, masked, and reconstructed images
        img, mask, recons = mae(img, vis=True)

        # Randomly select 5 slices to visualize
        for i in range(5):
            x, y, z, _, _ = img.shape
            x_c, y_c, z_c = random.randint(0, x - 1), random.randint(0, y - 1), random.randint(0, z - 1)

            # Create a figure for visualization
            plt.figure(figsize=(10, 10))

            # Plot raw image
            plt.subplot(1, 3, 1)
            plt.imshow(img[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Raw')

            # Plot masked image
            plt.subplot(1, 3, 2)
            plt.imshow(mask[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Masked')

            # Plot reconstructed image
            plt.subplot(1, 3, 3)
            plt.imshow(recons[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Reconstructed')

            # Save the figure
            plt.savefig(os.path.join(save_dir, f'recons_{iters}_{i}.png'), dpi=400, bbox_inches='tight')
            plt.close()

    # Clear GPU cache
    torch.cuda.empty_cache()


def visual_2d_att(learner, img, att, save_dir, iters):
    """
    Visualize the raw, masked, and reconstructed images in 2D with attention-based masking.

    Args:
        learner: The model used for reconstruction.
        img: Input image tensor.
        att: Attention map tensor.
        save_dir: Directory to save the visualization.
        iters: Current iteration number for naming the saved files.
    """
    with torch.no_grad():
        # Generate mask based on attention map
        mask, _ = learner.generate_mask_mam(att, step=iters, total_step=300000, alpha_t= 400000)

        # Get raw, masked, and reconstructed images
        img, mask, recons = learner(img, active_b1ff=mask, vis=True)

        # Randomly select 5 slices to visualize
        for i in range(5):
            x, y, z, _, _ = img.shape
            x_c, y_c, z_c = random.randint(0, x - 1), random.randint(0, y - 1), random.randint(0, z - 1)

            # Create a figure for visualization
            plt.figure(figsize=(10, 10))

            # Plot raw image
            plt.subplot(1, 3, 1)
            plt.imshow(img[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Raw')

            # Plot masked image
            plt.subplot(1, 3, 2)
            plt.imshow(mask[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Masked')

            # Plot reconstructed image
            plt.subplot(1, 3, 3)
            plt.imshow(recons[x_c, y_c, z_c].cpu().numpy(), cmap='gray')
            plt.title('Reconstructed')

            # Save the figure
            plt.savefig(os.path.join(save_dir, f'recons_{iters}_{i}.png'), dpi=400, bbox_inches='tight')
            plt.close()

    # Clear GPU cache
    torch.cuda.empty_cache()