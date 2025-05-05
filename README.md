# Image Colorization

> Transform black and white images into vibrant, colorized versions using deep learning.

This repository implements an end-to-end deep learning pipeline for automated image colorization using PyTorch. The system leverages Conditional Generative Adversarial Networks (cGANs) inspired by the Pix2Pix architecture to generate realistic colorization of grayscale images.

![Example of Image Colorization](assets/colorization-thumbnail.png)

## Overview

Image colorization is the process of taking grayscale (black and white) images and adding plausible colors to them. This project uses:

- **Conditional GANs**: A variant of GANs where the generator is conditioned on input data
- **U-Net Architecture**: For the generator, enabling precise pixel-level colorization
- **PatchGAN Discriminator**: For evaluating local patches of the image rather than the entire image
- **Lab Color Space**: Working in L\*a\*b\* color space for more natural colorization

## Key Features

- Train models from scratch or use pretrained weights
- Support for both custom generator and ResNet18-based U-Net
- Real-time visualization of training progress
- Fast inference on new grayscale images

## Project Requirements

- Python 3.6+
- PyTorch 1.11.0
- CUDA-capable GPU (recommended for training)

## Dependencies

Main libraries required:

```bash
torch==1.11.0
torchvision==0.12.0
fastai==2.4
Pillow==9.1.1
scikit-image==0.19.2
numpy==1.22.3
matplotlib==3.5.2
tqdm==4.64.0
```

A complete list of dependencies is available in `requirements.txt`.

## Getting Started

### Installation

1. Create a virtual environment (recommended)
2. Install the requirements:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

The project uses the COCO dataset by default, which will be automatically downloaded when running the training scripts. If you want to use your own dataset:

1. Prepare a folder with RGB images
2. Modify the data loading in `train.py` or `pretrained_model.py` to point to your dataset

## How It Works

### Architecture

This project implements two different generator architectures:

1. **Custom U-Net Generator**: Defined in `generator_model.py`, this is a from-scratch implementation with skip connections
2. **ResNet18-based U-Net**: Using a pretrained ResNet18 backbone, defined in `pretrained_generator.py`

Both generators take L channel (grayscale) as input and predict a and b channels.

The discriminator is a **PatchGAN** that focuses on classifying whether local image patches are real or fake.

### Color Space

Instead of working in RGB space, this model:

1. Converts RGB images to L\*a\*b\* color space
2. Uses the L channel (lightness) as input
3. Predicts the a\* and b\* channels (color information)
4. Recombines the channels to produce the final colorized image

This approach separates the lightness information from the color, allowing the model to focus solely on adding color.

## Training the Model

### Training from Scratch

To train the model from scratch using the custom U-Net generator:

```bash
python train.py
```

### Training with Pretrained Generator

For better results, you can use a pretrained ResNet18-based generator:

```bash
python pretrained_model.py
```

### Configuration

Model parameters can be adjusted in `config.py`:

```python
# Key parameters you might want to adjust
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
L1_LAMBDA = 100  # Weight of L1 loss
NUM_EPOCHS = 500
```

## Inference

To colorize your own grayscale images:

```bash
python test.py
```

Modify the paths in `test.py` to point to your test images.

## Results Visualization

During training, the model saves example outputs in either the `evaluations/` or `output/` directory, showing:
- Original grayscale image
- Generated colorized image
- Ground truth (real) colored image

This helps visualize the model's progress over time.

## Implementation Details

### Loss Functions

The model is trained using a combination of:

1. **Adversarial Loss**: Encourages the generator to create realistic colorizations that can fool the discriminator
2. **L1 Loss**: Ensures pixel-level accuracy compared to the ground truth, weighted by `L1_LAMBDA`

### Training Process

The training follows the standard GAN training procedure:
1. Update the discriminator to better distinguish real vs. fake colorized images
2. Update the generator to better fool the discriminator while staying close to ground truth

```python
# Simplified training loop
# Train Discriminator
d_real = disc(real_image)
d_fake = disc(fake_image.detach())
d_loss = (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))) / 2

# Train Generator
D_fake = disc(fake_image)
G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
L1_loss = L1(fake_color, ab) * L1_LAMBDA
G_loss = G_fake_loss + L1_loss
```

## Project Structure

```
├── config.py                  # Configuration parameters
├── dataset.py                 # Dataset and dataloader definitions
├── generator_model.py         # Custom U-Net generator
├── PatchDiscriminator_model.py # PatchGAN discriminator
├── pretrained_generator.py    # ResNet18-based generator
├── pretrained_model.py        # Training with pretrained generator
├── test.py                    # Inference script
├── train.py                   # Training script
└── utils.py                   # Utility functions
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

- The Pix2Pix paper that inspired this implementation: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- The FastAI library for providing pretrained models
