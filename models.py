import torch
import torch.nn as nn
import torch.nn.functional as F

# Mapping Network: Transforms a latent vector into a higher-dimensional space using multiple layers
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),  # Linear layer to transform latent vector
                nn.LeakyReLU(0.2)  # Leaky ReLU activation function
            ])
        self.mapping = nn.Sequential(*layers)  # Sequential container to chain layers

    def forward(self, z):
        return self.mapping(z)  # Forward pass through the mapping network

# Adaptive Instance Normalization (AdaIN): Normalizes the input and applies style modulation
class AdaIN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels)  # Instance normalization layer
        self.style_scale = nn.Linear(latent_dim, in_channels)  # Linear layer to generate scale
        self.style_bias = nn.Linear(latent_dim, in_channels)  # Linear layer to generate bias

    def forward(self, x, w):
        x_norm = self.norm(x)  # Normalize input features
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)  # Compute scale factor
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)  # Compute bias factor
        return scale * x_norm + bias  # Apply scale and bias to normalized input

# Modulated Convolutional Layer: Convolutional layer with modulation and optional demodulation
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, demodulate=True, use_adain=False):
        super(ModulatedConv2d, self).__init__()
        self.demodulate = demodulate  # Whether to apply demodulation
        self.kernel_size = kernel_size
        self.use_adain = use_adain  # Whether to use AdaIN
        self.eps = 1e-8  # Small epsilon to avoid division by zero

        # Initialize weights with a small random value
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)

        # Modulation network to adapt the weights based on the latent vector
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, in_channels),
            nn.LayerNorm(in_channels),  # Normalize the input to the modulation network
            nn.LeakyReLU(0.2)  # Leaky ReLU activation function
        )

        # AdaIN layer if specified
        if self.use_adain:
            self.adain = AdaIN(in_channels, latent_dim)

        # Scaling factor for modulation
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # Padding to maintain spatial dimensions after convolution
        self.padding = kernel_size // 2

    def forward(self, x, w):
        b = x.size(0)  # Batch size

        # Compute modulation with scaling
        s = self.modulation(w).view(b, 1, -1, 1, 1) * self.scale
        weight = self.weight.unsqueeze(0) * s

        # Apply AdaIN if enabled
        if self.use_adain:
            x = self.adain(x, w)

        # Apply demodulation if enabled
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * d.view(b, -1, 1, 1, 1)

        # Reshape weight for convolution
        weight = weight.view(b * weight.size(1), *weight.size()[2:])

        # Reshape input x and apply convolution
        x = x.view(1, b * x.size(1), x.size(2), x.size(3))
        x = F.conv2d(x, weight, padding=self.padding, groups=b)
        x = x.view(b, -1, x.size(2), x.size(3))

        return x

# Synthesis Block: A building block in the synthesis network, includes multiple modulated convolutions
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, use_adain=True):
        super(SynthesisBlock, self).__init__()
        # Sequence of modulated convolutions followed by an activation function
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv3 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv4 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv5 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.activation = nn.LeakyReLU(0.2)  # Leaky ReLU activation function
        # Final convolution to map to RGB space
        self.to_rgb = ModulatedConv2d(out_channels, 3, 1, latent_dim, demodulate=False, use_adain=use_adain)

        # Upsampling layer to increase spatial resolution
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Skip connection to add the previous layer's output
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, w):
        x = self.upsample(x)  # Upsample the input
        skip_x = self.skip(x)  # Compute skip connection
        x = self.activation(self.conv1(x, w))  # Apply first convolution and activation
        x = self.activation(self.conv2(x, w))  # Apply second convolution and activation
        x = self.activation(self.conv3(x, w))  # Apply third convolution and activation
        x = self.activation(self.conv4(x, w))  # Apply fourth convolution and activation
        x = self.activation(self.conv5(x, w))  # Apply fifth convolution and activation
        x += skip_x  # Add skip connection
        rgb = self.to_rgb(x, w)  # Generate RGB image
        return x, rgb

# Synthesis Network: Constructs the final image from the latent vector by stacking synthesis blocks
class SynthesisNetwork(nn.Module):
    def __init__(self, latent_dim, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        # Initial constant input to start the synthesis
        self.const_input = nn.Parameter(torch.randn(1, latent_dim, 4, 4))
        self.blocks = nn.ModuleList()
        # Add initial synthesis block
        self.blocks.append(SynthesisBlock(latent_dim, latent_dim, latent_dim))

        # Add subsequent synthesis blocks with decreasing channel size
        for i in range(1, num_blocks):
            in_channels = latent_dim // (2 ** (i - 1))
            out_channels = latent_dim // (2 ** i)
            self.blocks.append(SynthesisBlock(in_channels, out_channels, latent_dim))

    def forward(self, w):
        x = self.const_input.repeat(w.shape[0], 1, 1, 1)  # Initialize with constant input
        rgb = None
        for i, block in enumerate(self.blocks):
            x, rgb_new = block(x, w[:, i])  # Forward pass through each block
            rgb = rgb_new if rgb is None else self.upsample(rgb) + rgb_new  # Accumulate RGB output
        return rgb

    def upsample(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample the image

# StyleGAN Generator: Combines the mapping and synthesis networks to generate images from latent vectors
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, num_mapping_layers, num_synthesis_blocks):
        super().__init__()
        self.mapping_network = MappingNetwork(latent_dim, num_mapping_layers)  # Initialize mapping network
        self.synthesis_network = SynthesisNetwork(latent_dim, num_synthesis_blocks)  # Initialize synthesis network

    def forward(self, z):
        w = self.mapping_network(z)  # Map latent vector to intermediate space
        w = w.unsqueeze(1).repeat(1, self.synthesis_network.num_blocks, 1)  # Repeat for each synthesis block
        img = self.synthesis_network(w)  # Generate image from the latent vector
        return img

# Minibatch Discrimination Layer: Encourages the model to differentiate between different instances in a batch
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))  # Learnable matrix for minibatch discrimination

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)  # Flatten the input
        M = x_flat.matmul(self.T.view(self.T.shape[0], -1))  # Compute the minibatch features
        M = M.view(-1, self.T.shape[1], self.T.shape[2])  # Reshape features for pairwise comparison
        M_i = M.unsqueeze(0)  # Expand dimensions for pairwise comparison
        M_j = M.unsqueeze(1)
        abs_diff = torch.abs(M_i - M_j).sum(2)  # Compute pairwise differences
        features = torch.exp(-abs_diff)  # Apply exponential to differences
        return torch.cat([x_flat, features.sum(0)], 1)  # Concatenate original and minibatch features

# Discriminator: Classifies images as real or fake and extracts intermediate features for analysis
class Discriminator(nn.Module):
    def __init__(self, img_resolution, base_channels=64):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = base_channels

        # Initial convolutional block
        self.blocks.append(nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1),  # Convolution to reduce image size
            nn.LeakyReLU(0.2)  # Activation function
        ))

        current_resolution = img_resolution // 2

        # Subsequent convolutional blocks with increasing channels
        while current_resolution > 4:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ))
            channels *= 2
            current_resolution //= 2

        # Final convolutional block and flattening
        self.blocks.append(nn.Sequential(
            nn.Conv2d(channels, channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Flatten()  # Flatten the output
        ))

        self.final_conv_size = channels
        # Add minibatch discrimination to the discriminator
        self.minibatch_discrimination = MinibatchDiscrimination(self.final_conv_size, 100, int(self.final_conv_size * 0.2))
        self.fc = nn.Linear(self.final_conv_size + int(self.final_conv_size * 0.2), 1)  # Final classification layer

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)  # Forward pass through each block
            if return_features:
                features.append(x)  # Collect features if requested
        x = self.minibatch_discrimination(x)  # Apply minibatch discrimination
        x = self.fc(x)  # Final classification
        if return_features:
            return x.view(-1), features  # Return both classification and features
        return x.view(-1)  # Return classification only
