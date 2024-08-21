import torch
import torch.nn as nn
import torch.nn.functional as F

# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2)
            ])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
    
# Adaptive Instance Normalization (AdaIN)
class AdaIN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.style_scale = nn.Linear(latent_dim, in_channels)
        self.style_bias = nn.Linear(latent_dim, in_channels)

    def forward(self, x, w):
        # Normalize the input
        x_norm = self.norm(x)
        # Get the style scale and bias from the latent vector
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        # Apply scale and bias
        return scale * x_norm + bias

# Modulated Convolutional Layer
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_dim, demodulate=True, use_adain=False):
        super(ModulatedConv2d, self).__init__()
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.use_adain = use_adain
        self.eps = 1e-8

        # Initialize weight parameter with a better initialization method
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02)

        # Modulation layer with LayerNorm
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, in_channels),
            nn.LayerNorm(in_channels),
            nn.LeakyReLU(0.2)
        )

        # AdaIN layer
        if self.use_adain:
            self.adain = AdaIN(in_channels, latent_dim)

        # Learnable scaling factor for modulation
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # Calculate padding to keep spatial dimensions the same
        self.padding = kernel_size // 2

    def forward(self, x, w):
        b = x.size(0)  # Batch size

        # Modulation with scaling factor
        s = self.modulation(w).view(b, 1, -1, 1, 1) * self.scale
        weight = self.weight.unsqueeze(0) * s

        # Apply AdaIN if enabled
        if self.use_adain:
            x = self.adain(x, w)

        # Demodulation (optional)
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

# Synthesis Block
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, use_adain=True):
        super(SynthesisBlock, self).__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv3 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv4 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.conv5 = ModulatedConv2d(out_channels, out_channels, 3, latent_dim, use_adain=use_adain)
        self.activation = nn.LeakyReLU(0.2)
        self.to_rgb = ModulatedConv2d(out_channels, 3, 1, latent_dim, demodulate=False, use_adain=use_adain)

        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, w):
        x = self.upsample(x)
        skip_x = self.skip(x)
        x = self.activation(self.conv1(x, w))
        x = self.activation(self.conv2(x, w))
        x = self.activation(self.conv3(x, w))
        x = self.activation(self.conv4(x, w))
        x = self.activation(self.conv5(x, w))
        x += skip_x  # Add skip connection
        rgb = self.to_rgb(x, w)
        return x, rgb

# Synthesis Network
class SynthesisNetwork(nn.Module):
    def __init__(self, latent_dim, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self.const_input = nn.Parameter(torch.randn(1, latent_dim, 4, 4))
        self.blocks = nn.ModuleList()
        self.blocks.append(SynthesisBlock(latent_dim, latent_dim, latent_dim))

        for i in range(1, num_blocks):
            in_channels = latent_dim // (2 ** (i - 1))
            out_channels = latent_dim // (2 ** i)
            self.blocks.append(SynthesisBlock(in_channels, out_channels, latent_dim))

    def forward(self, w):
        x = self.const_input.repeat(w.shape[0], 1, 1, 1)
        rgb = None
        for i, block in enumerate(self.blocks):
            x, rgb_new = block(x, w[:, i])
            rgb = rgb_new if rgb is None else self.upsample(rgb) + rgb_new
        return rgb

    def upsample(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

# Generator
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, num_mapping_layers, num_synthesis_blocks):
        super().__init__()
        self.mapping_network = MappingNetwork(latent_dim, num_mapping_layers)
        self.synthesis_network = SynthesisNetwork(latent_dim, num_synthesis_blocks)

    def forward(self, z):
        w = self.mapping_network(z)
        w = w.unsqueeze(1).repeat(1, self.synthesis_network.num_blocks, 1)
        img = self.synthesis_network(w)
        return img

# Minibatch Discrimination Layer
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)  # Flatten the input
        M = x_flat.matmul(self.T.view(self.T.shape[0], -1))
        M = M.view(-1, self.T.shape[1], self.T.shape[2])
        M_i = M.unsqueeze(0)
        M_j = M.unsqueeze(1)
        abs_diff = torch.abs(M_i - M_j).sum(2)
        features = torch.exp(-abs_diff)
        return torch.cat([x_flat, features.sum(0)], 1)

# Modify the Discriminator to return intermediate features
class Discriminator(nn.Module):
    def __init__(self, img_resolution, base_channels=64):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = base_channels

        self.blocks.append(nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1),
            nn.LeakyReLU(0.2)
        ))

        current_resolution = img_resolution // 2

        while current_resolution > 4:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 4, 2, 1),
                nn.LeakyReLU(0.2)
            ))
            channels *= 2
            current_resolution //= 2

        self.blocks.append(nn.Sequential(
            nn.Conv2d(channels, channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        ))

        self.final_conv_size = channels
        self.minibatch_discrimination = MinibatchDiscrimination(self.final_conv_size, 100, int(self.final_conv_size * 0.2))
        self.fc = nn.Linear(self.final_conv_size + int(self.final_conv_size * 0.2), 1)

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        x = self.minibatch_discrimination(x)
        x = self.fc(x)
        if return_features:
            return x.view(-1), features
        return x.view(-1)  # Flatten to a 1D tensor