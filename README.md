# OrgMIM

Datasets, codes, and pretrained weights for **“Masked Image Modeling for Generalizable Organelle Segmentation in Volume EM”** *(under review)*.

## 🗂️ Pretraining Database: IsoOrg-1K

We introduce **IsoOrg-1K**, a diverse organelle-specific dataset collected from [OpenOrganelle](https://openorganelle.janelia.org/). Detailed information is shown below. The complete dataset and generated **membrane maps** are available [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).
Meanwhile, we are actively curating and integrating organelle datasets from other platforms, and will continue to update this repository to support larger-scale pre-training in the future.
<!-- ![Dataset Details](./Figures/Details.jpg) -->

## 📊 Downstream Segmentation Datasets

We conduct extensive experiments on four representative datasets with varying voxel resolutions and biological contexts. The processed and partitioned data can be downloaded from [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

## ⚙️ Environments

The complete Conda environment has been packaged for direct use. You can download and unzip it from [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

## 🔬 Pretraining with OrgMIM

### Generation of membrane maps

### Dual-branch masked image modeling

## 📉 Downstream Fine-tuning
### Pretrianed weights transfer on STU-Net
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
### Pretrianed weights transfer on UNETR
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
| MAE-based OrgMIM (Tiny)    | orgmim_mae_t_learner.ckpt     | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |
| Spark-based OrgMIM (Tiny)  | orgmim_spark_t_learner.ckpt   | [Hugging Face](https://huggingface.co/yanchaoz/OrgMIM)       |

## 🙏 Acknowledgements

We thank all contributors and open-source dataset providers for their support in this project.
