# GenCompress: Generative Latent Diffusion for Efficient Spatiotemporal Data Reduction

## Overview

### Main Contributions

1. **Keyframe Compressor (𝐶1)**: Compresses each keyframe into a latent space.
2. **Latent Diffusion Model (𝐶2)**: Generates other latent representations from the compressed keyframe latent space.
3. **Error-bound Guarantee (𝐶3)**: Applies a post-processing step to ensure the user-defined error bound is met.

### Computational Artifacts

- **Artifact 1 (𝐴1)**: Performance comparison on the E3SM dataset.
- **Artifact 2 (𝐴2)**: Performance comparison on the S3D dataset.
- **Artifact 3 (𝐴3)**: Performance comparison on the JHTDB dataset.
- **Artifact 4 (𝐴4)**: Inference speed for encoding and decoding.

### Artifact Setup

#### Hardware
All experiments were conducted on a system equipped with:
- NVIDIA A100 GPU
- 4-core CPU

#### Software
- torch==2.2.0
- torchvision==0.15.0
- einops==0.6.0
- einops-exts==0.0.1
- rotary-embedding-torch==0.2.0
- numpy==1.24.0
- Pillow==9.5.0
- tqdm==4.64.1
- zstandard==0.21.0
- collections==3.0.1
- huffman==0.4.0
- scikit-learn==1.2.1

#### Datasets / Inputs
- **E3SM**: Spatiotemporal climate simulation data , available [here](https://link_to_s3d_dataset).
- **S3D**: Scientific combustion simulation dataset, available [here](https://link_to_s3d_dataset).
- **JHTDB**: High-fidelity turbulence simulation dataset, available [here](https://link_to_jhtdb_dataset).

### Installation and Deployment

To install the required Python packages and dependencies, run the following command:

```bash
pip install -r requirements.txt
```

#### Model Training

To train the keyframe compressor, use the following command:

```bash
bash train_model2d.sh
```

To train the latent diffusion model, use the following command:

```bash
bash train_model.sh
```

#### Model Evaluation
evaluate.ipynb

