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

### Expected Reproduction Time

- **Keyframe Compressor Training (𝐶1)**: 480 minutes on an NVIDIA A100 GPU.
- **Latent Diffusion Model Training (𝐶2)**: 1200 minutes on an NVIDIA A100 GPU.
- **Model Inference**: Approximately 5 minutes.

### Artifact Setup

#### Hardware
All experiments were conducted on a system equipped with:
- NVIDIA A100 GPU
- 4-core CPU

#### Software
- **PyTorch 2.2.0**

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

