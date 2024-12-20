# Official PyTorch implementation of `Enhancing 3D MRI Analysis with Generative Adversarial Networks`
This repo contains implementation for the 3D MRI Reconstruction and Representation Learning task.

[Overview](/overview.png)

## Implementation details
This code was implemented with PyTorch 2.1, dependencies could be installed from `requirements.txt`

- run `pip install requirements.txt` in your conda environment before running inference

- You can understand training details from `train_0x.py` and supporting `Models.py` files.

- `outs.py` contains the details regarding downstream classification which uses representations generated using `embed.py` files for respective methods (also BYOL and SimCLR).

We also provide our implementations of `vit_ae`, `BYOL` and `SimCLR` used for the experiments.

## Datasets and processing
- We use AOMIC, ADNI and PPMI public datasets and a private COVID dataset
- First, we perform skull stripping as discussed in paper, followed by pythonic operations to resize 3D images to (64 * 64 * 64) (these operations can be found in the code in the Dataset class for loading examples)

