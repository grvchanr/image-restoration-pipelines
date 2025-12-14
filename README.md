# Image Restoration Pipelines

This project explores multiple approaches to image restoration, starting from
classical image processing techniques to deep learning–based models.

## Implemented Methods
- Edge-based sketch generation using OpenCV
- Denoising autoencoder for image reconstruction
- Comparison of reconstruction losses (MSE vs L1)
- Conditional GAN (Pix2Pix-style) for image denoising

## Key Observations
- MSE loss produces smoother but blurrier reconstructions
- L1 loss preserves sharper edges and structure
- Pure adversarial loss in GANs is unstable
- Combining GAN loss with L1 stabilizes training

## Dataset
- FashionMNIST

## Tech Stack
- PyTorch
- OpenCV
- Matplotlib

## Results
See the `results/` folder for visual outputs.
