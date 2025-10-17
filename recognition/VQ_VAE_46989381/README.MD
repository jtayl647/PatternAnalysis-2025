# HipMRI — Prostate Cancer Generative Model (VQVAE2)
**Generative modelling of HipMRI prostate-cancer 2D slices using VQ-VAE**

**Goal:** produce visually clear reconstructions and generations with Structural Similarity (SSIM) > 0.6.

## Motivation
Medical imaging datasets, such as prostate MRI scans, are often limited in size and diversity due to patient privacy concerns, acquisition costs, and labeling complexity. This scarcity makes it difficult to train robust deep learning models for diagnosis or segmentation. Generative models like VQ-VAE-2 offer a solution by learning compact and meaningful latent representations of MRI data, enabling the generation of realistic synthetic images and high-quality reconstructions. Such models can enhance dataset diversity, support data augmentation, and improve the understanding of structural patterns associated with prostate cancer.

## Algorithm Description
A VQ-VAE is a variational autoencoder–style model that uses a discrete codebook to represent the latent space [2]. This allows it to achieve GAN-like realism while maintaining the stable, monotonic convergence of a VAE, avoiding the instability of adversarial training. VQ-VAE-2 extends this approach with a hierarchical latent structure to capture both global and fine-grained image features [1].

The VQ-VAE-2 model consists of a hierarchical encoder–decoder structure with two levels of latent representations. The **bottom encoder** extracts fine-grained features, while the **top encoder**, applied after downsampling, captures global features. Both latents are discretized using **vector quantizers**: the top quantizer processes the top-level latent, and the bottom quantizer processes a combination of the bottom-level latent and the upsampled top-level latent. The **decoder** then reconstructs the image from the quantized latent using transposed convolutions, residual blocks, instance normalization, and dropout. Residual connections and instance normalization throughout the network help maintain stable training and preserve high-quality feature representations.

![VQ-VAE2 Model](images/vqvae2_blocks.png)
*Figure: Overview of the VQ-VAE-2 hierarchical architecture, showing the bottom and top encoders, vector quantizers, and decoder [1].*

## Dependencies
It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.


## References
1. Nakada, K., & Kimata, H. (2023). A Study on Voxel Shape Generation and Reconstruction with VQ-VAE-2. A Study on Voxel Shape Generation and Reconstruction with VQ-VAE-2, 9–15. https://doi.org/10.1145/3606283.3606285

2. Oord, A. van den, Vinyals, O., & Kavukcuoglu, K. (2018). Neural Discrete Representation Learning. ArXiv:1711.00937 [Cs]. https://arxiv.org/abs/1711.00937