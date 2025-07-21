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

# Move to GPU if available
if torch.cuda.is_available():
    self.network.cuda()

# Load pretrained model weights
saved_model = torch.load('orgmim_spark_b_learner.ckpt')
pretrained_dict = saved_model['model_weights']
print(pretrained_dict.keys())

# Process and load encoder weights into the current model
new_file3 = OrderedDict()
for old_key, value in pretrained_dict.items():
    if 'encoder' in old_key:
        new_key = old_key.split('sp_cnn.')[-1]
        new_file3[new_key] = value

self.network.load_state_dict(new_file3, strict=False)

# Check loaded convolution block layers
mod_dict = self.network.state_dict()
for key, _ in mod_dict.items():
    if 'conv_blocks' in key:
        if key in new_file3 and mod_dict[key].shape == new_file3[key].shape:
            print('This layer worked:', key)

```


### nnUNet-based segmentation

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
