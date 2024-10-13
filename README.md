# Official PyTorch implementation of `Enhancing 3D MRI Analysis with Generative Adversarial Networks`
This repo contains implementation for the 3D MRI Reconstruction and Representation Learning task.

## Implementation details
This code was implemented with PyTorch 2.1, dependencies could be installed from `requirements.txt`

- run `pip install requirements.txt` in your conda environment before running inference

- To run inference (`infer.ipynb`) download the pre-trained weights from this [link](add_a_link_here). You can understand training details from `train_0x.py` and supporting `Models.py` files.

- `outs.py` contains the details regarding downstream classification which uses representations generated using `embed.py` files for respective methods (also BYOL and SimCLR).

We also provide our implementations of `vit_ae`, `BYOL` and `SimCLR` used for the experiments.

## Paper
For datasets and other details, refer our [pre-print](add_a_link_here)
