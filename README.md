# StyleGAN Flower Image Generator

## Introduction

This project implements a StyleGAN-based Generative Adversarial Network (GAN) for generating high-quality images of flowers. StyleGAN, developed by NVIDIA, is known for its ability to generate highly realistic images by learning both the style and structure of a target dataset. In this project, we use StyleGAN to train on a flower image dataset, enabling the generation of new, photorealistic flower images.

## Minimal Scripts Overview

This repository contains the following key scripts:

1. **models.py**: Defines the neural network architectures used in the GAN, including the **generator** and **discriminator** models. The generator creates synthetic images from random noise, while the discriminator evaluates them against real images.

2. **utils.py**: Contains utility functions for data preprocessing, model checkpointing, and image generation. It also includes the `GANTrainer` class, which manages the entire training process, including loading data, optimizing the models, and saving outputs.

3. **main.py**: The main script to start the training process. It initializes the GAN model, prepares the dataset, and begins training based on the parameters defined in `utils.py`.

## Summary of Model Components

### 1. MappingNetwork
The **MappingNetwork** class transforms a latent vector (`z`) into a higher-dimensional space using multiple linear layers and LeakyReLU activation functions. This mapping helps in converting a low-dimensional latent space into a space where features are more meaningful and separable for image generation.

### 2. AdaIN (Adaptive Instance Normalization)
The **AdaIN** layer normalizes input features and applies style modulation based on a latent vector. It combines instance normalization with learned affine transformations (scale and bias) derived from the latent vector, allowing control over style characteristics in generated images.

### 3. ModulatedConv2d
The **ModulatedConv2d** class implements a modulated convolutional layer with optional demodulation and AdaIN. This layer modulates convolution weights according to a latent vector, and optionally applies AdaIN for style modulation. It can also demodulate the weights to normalize feature magnitudes across channels, ensuring consistency in feature activations.

### 4. SynthesisBlock
A **SynthesisBlock** is composed of multiple modulated convolutional layers. It forms a part of the image synthesis network, applying several convolutions and activations in sequence to progressively generate and refine features, contributing to the construction of an image.

### 5. SynthesisNetwork
The **SynthesisNetwork** consists of multiple synthesis blocks arranged in a sequence to generate an image from a latent vector. Starting with a constant input, it applies successive synthesis blocks to upsample and refine the image, progressively increasing the level of detail and realism.

### 6. StyleGANGenerator
The **StyleGANGenerator** combines a mapping network and a synthesis network to generate images from latent vectors. It first transforms the latent vector into an intermediate latent space using the mapping network. The transformed vector is then used by the synthesis network to generate the final image, enabling control over the style and attributes of the generated images.

### 7. MinibatchDiscrimination
The **MinibatchDiscrimination** layer helps the model distinguish between different instances within a batch, encouraging diversity in the outputs and preventing mode collapse. By comparing features across a batch, it promotes more varied and realistic image generation by adding diversity-promoting signals to the discriminator’s outputs.

### 8. Discriminator
The **Discriminator** network classifies images as real or fake and extracts intermediate features for analysis. It is composed of multiple convolutional blocks with dropout layers and a minibatch discrimination layer. This structure helps the discriminator to become robust against overfitting and promotes the generation of diverse outputs. The network concludes with a fully connected layer for binary classification (real or fake).

## Installation

To set up this project, follow these steps:

1. **Clone the repository**:

```
git clone https://github.com/yourusername/stylegan-flower-generator.git
cd stylegan-flower-generator
```
2. **Install the required dependencies**:

This project requires Python 3.7+ and the following Python libraries:

- PyTorch (with CUDA support for GPU acceleration)
- torchvision
- tqdm

You can install these dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

To train the StyleGAN model on flower dataset, run the `main.py` script. This script will start the training process using the parameters defined in `utils.py`.

You can run main script like below:

```
python main.py
```

You should configure `batch_size` according to your GPU's RAM size to prevent overflow.

## Generating Images

Models after train produce a fake dataset and example of generated images in `generated_images` directory. Also while training process saves generated images of batchs in mentioned directory in intervals.

## Conclusion

By following these steps, you should be able to train a StyleGAN model on a flower dataset and generate new, realistic flower images. For more information on the project, please check my story on medium website. This is the [link](https://medium.com/@a.r.amouzad.m/stylegan-for-creating-flower-images-d29ac8391f7e).

Also if you want trained model you can find it Kaggle. This is the [link](https://www.kaggle.com/models/alirezaamouzad/stylegan_flower_128x128).
