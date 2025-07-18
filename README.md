# OrgMIM

Datasets, codes, and pretrained weights for **“Masked Image Modeling for Generalizable Organelle Segmentation in Volume EM”** *(under review)*.

## 🗂️ Pretraining Database: IsoOrg-1K

We introduce **IsoOrg-1K**, a diverse organelle-specific dataset collected from [OpenOrganelle](https://openorganelle.janelia.org/). Detailed information is shown below. The complete dataset and generated **membrane maps** are available [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

<!-- ![Dataset Details](./Figures/Details.jpg) -->

## ⚙️ Environments

The complete Conda environment has been packaged for direct use. You can download and unzip it from [here](https://huggingface.co/datasets/yanchaoz/IsoOrg-1K).

## 🔬 Pretraining with OrgMIM

### Generation of membrane maps

### Dual-branch masked image modeling

## 📉 Downstream Fine-tuning

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
