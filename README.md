# OrgMIM

Datasets, codes, and pretrained weights for **“Masked Image Modeling for Generalizable Organelle Segmentation in Volume EM”** *(under review)*

## 🗂️ Pretraining Database: IsoOrg-1K

We introduce **IsoOrg-1K**, a diverse organelle-specific dataset collected from [OpenOrganelle](https://openorganelle.janelia.org/). Detailed information is shown below. The complete dataset and generated **membrane maps** are available [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).
Meanwhile, we are actively curating and integrating organelle datasets from other platforms, and will continue to update this repository to support larger-scale pretraining in the future.
<!-- ![Dataset Details](./Figures/Details.jpg) -->

## 📊 Downstream Segmentation Datasets

We conduct extensive experiments on four representative datasets with varying voxel resolutions and biological contexts. The processed and partitioned data can be downloaded from [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

## ⚙️ Environments

The complete Conda environment has been packaged for direct use. You can download and unzip it from [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

## 🔬 Pretraining with OrgMIM

### Generation of membrane attention maps
#### Step 1. Loading a Visual Foundation Model
First, install the [Segment Anything](https://github.com/facebookresearch/segment-anything) package:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Then, load the SAM model and weights in Python:
```python
from segment_anything import sam_model_registry, SamPredictor

# Available model types: "vit_h", "vit_l", "vit_b"
model_type = "vit_h"
checkpoint_path = "sam_vit_h_4b8939.pth"

# Download the checkpoint from the official GitHub:
# https://github.com/facebookresearch/segment-anything#model-checkpoints

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

```
#### Step 2. Pixel-level Similarity Calcuation
```python
# Load a single-channel TIFF image and convert it to 3-channel RGB
img = tiff[i, :, :]
img_rgb = np.stack([img] * 3, axis=0)  # Shape: (3, H, W)
image = np.transpose(img_rgb, (1, 2, 0))  # Shape: (H, W, 3)

# Initialize SAM predictor and extract features
predictor = SamPredictor(sam)
predictor.set_image(image)
embedding = predictor.features
embedding = embedding.detach().cpu().numpy().squeeze()  # Shape: (C, H, W)

# Compute pixel affinities from embeddings
affs = embeddings_to_affinities(embedding, delta_v=0.5, delta_d=1.5)
affs = np.minimum(affs[0], affs[1])  # Take element-wise min of first two channels
affs = affs[1:, 1:]

# Resize affinity map to desired shape
affs_resized = nearest_neighbor_resize(affs, (512, 512))

# Convert affinity map to uint8 format for saving or visualization
affs_uint8 = np.uint8(255 * affs_resized)
```

| Function / Class             | Defined In           | Description                                      |
|-----------------------------|----------------------|--------------------------------------------------|
| `embeddings_to_affinities`  | `mam_utils.py`  | Converts pixel embeddings into affinity maps     |
| `nearest_neighbor_resize`   | `mam_utils.py`     | Resizes 2D arrays using nearest neighbor interpolation |

### Dual-branch masked image modeling

After downloading the dataset, simply run the following script to start training OrgMIM:

```bash
python dual_branch_mim.py --config configs/orgmim.yaml
```
To run it in distributed mode (e.g., on 4 GPUs):
```bash
torchrun --nproc_per_node=4 dual_branch_mim_dp.py --config configs/orgmim.yaml
```

## 📉 Downstream Fine-tuning
All downstream fine-tuning experiments were conducted within the nnU-Net framework. Notably, the input data were normalized by dividing pixel intensities by **255.0**.

### Pretrianed weights transfer on STU-Net (CNN-based)
```python
import torch
from collections import OrderedDict

# Initialize network
self.network = STUNet(
    self.num_input_channels,
    self.num_classes,
    depth=[1, 1, 1, 1, 1, 1],
    dims=[32, 64, 128, 256, 512, 512],
    pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
    conv_kernel_sizes=self.net_conv_kernel_sizes
)

# Load pretrained model weights
saved_model = torch.load('/***/***/orgmim_spark_b_learner.ckpt')
pretrained_dict = saved_model['model_weights']

# Process and load encoder weights into the current model
new_dict = OrderedDict()
for old_key, value in pretrained_dict.items():
    if 'encoder' in old_key:
        new_key = old_key.split('sp_cnn.')[-1]
        new_dict[new_key] = value

self.network.load_state_dict(new_dict, strict=False)
```
### Pretrianed weights transfer on UNETR (ViT-based)
```python
# Initialize network
self.network =  UNETR(
                in_channels=self.num_input_channels,
                out_channels=self.num_classes,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                norm_name='instance',
                conv_block=True,
                res_block=True,
                kernel_size=3,
                skip_connection=False,
                show_feature=False,
                dropout_rate=0.0)

# Load pretrained model weights
saved_model = torch.load('/***/***/orgmim_mae_b_learner.ckpt')
vit_state_dict = checkpoint['model_weights']
self.network.vit.load_state_dict(vit_state_dict, strict=False)
```
## 🎨 Visual Results

### Mask reconstruction by directly loading the MIM learner

### PCA visualization of dense embeddings from the vision foundation models

## 💾 Released Weights

| Methods                   | Models                        | Download                                                      |
|----------------------------|-------------------------------|---------------------------------------------------------------|
| MAE-based OrgMIM (Base)    | orgmim_mae_b_learner.ckpt     | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| Spark-based OrgMIM (Base)  | orgmim_spark_b_learner.ckpt   | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| MAE-based OrgMIM (Large)   | orgmim_mae_l_learner.ckpt     | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| Spark-based OrgMIM (Large) | orgmim_spark_l_learner.ckpt   | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| MAE-based OrgMIM (Small)    | orgmim_mae_s_learner.ckpt     | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| Spark-based OrgMIM (Small)  | orgmim_spark_s_learner.ckpt   | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |

## 🙏 Acknowledgements

We thank all contributors and open-source dataset providers for their support in this project.
