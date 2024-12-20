# Bootstrap Your Own Latent [BYOL](https://arxiv.org/abs/2006.07733) implementation and experiments for 3D_MRI_

BYOL is a self-supervised method, highly similar to current contrastive learning methods, without the need for negative samples. We incorporated BYOL as a baseline for learning representations of 3D MRI images. We initialize BYOL training with video resnet weights on initial layers, as opposed to random initialization on our method.

For official implementation of BYOL, visit the github repository: [BYOL-deepmind](https://github.com/google-deepmind/deepmind-research/tree/master/byol)

## Install requirements
To install the needed requirements in a new conda environment (BYOL) use

```bash
conda env create -f environment.yml
```

## Usage
- Parameters required: The `csv` file as described [here](https://github.com/sujeet1024/3D_MRI_/blob/main/README.md)
- Simply run the [run.py] file in the conda environment to start training. 
- For inference initialize BYOL module in eval mode as in [embeds.py], this file is used to generate embeddings for downstream classification task.