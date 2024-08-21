import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from models import StyleGANGenerator, Discriminator

torch.cuda.empty_cache()

# Function to calculate gradient norm
def calculate_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_latest_model_epoch(model_dir, model_type):
    model_files = os.listdir(model_dir)
    model_files = [f for f in model_files if f.startswith(model_type)]
    if not model_files:
        return None, 0

    epochs = [int(re.search(r'\d+', f).group()) for f in model_files]
    latest_epoch = max(epochs)
    latest_model_file = f'{model_type}_epoch_{latest_epoch}.pth'
    return os.path.join(model_dir, latest_model_file), latest_epoch

def generate_fake_images(generator_path, output_dir, latent_dim=512, num_mapping_layers=10, num_images=256, batch_size=64, img_resolution=128):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, 'fake_dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # Load the trained generator model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = StyleGANGenerator(latent_dim, num_mapping_layers, num_synthesis_blocks=int(torch.log2(torch.tensor(img_resolution))) - 2).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()  # Set the generator to evaluation mode

    num_batches = (num_images + batch_size - 1) // batch_size
    all_images = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Generate random noise
            noise = torch.randn(batch_size, latent_dim, device=device)
            # Generate fake images
            fake_images = generator(noise)
            
            # Save individual images
            for i, image in enumerate(fake_images):
                img_path = os.path.join(dataset_dir, f'fake_image_{batch_idx * batch_size + i + 1}.png')
                save_image(image, img_path, normalize=True)
                all_images.append(image)
            
            print(f'Generated and saved batch {batch_idx + 1}/{num_batches}')

    # Concatenate all images into a single tensor
    all_images = torch.stack(all_images)
    # Create a grid of images
    grid = make_grid(all_images, nrow=int(torch.sqrt(torch.tensor(num_images))), normalize=True)

    # Save the grid image
    save_image(grid, os.path.join(output_dir, 'fake_images_grid.png'), normalize=True)
    print('Saved the image grid as fake_images_grid.png')

def train_discriminator(netD, netG, criterion, optimizerD, real_images, latent_dim, device, max_grad_norm_d):
    netD.zero_grad()
    b_size = real_images.size(0)
    label = torch.full((b_size,), 1, dtype=torch.float, device=device)

    # Forward pass on real images
    output = netD(real_images)
    lossD_real = criterion(output, label)
    lossD_real.backward()

    # Generate fake images and forward pass on them
    noise = torch.randn(b_size, latent_dim, device=device)
    fake_images = netG(noise)
    label.fill_(0)
    output = netD(fake_images.detach())
    lossD_fake = criterion(output, label)
    lossD_fake.backward()

    lossD = lossD_real + lossD_fake

    # Gradient clipping for Discriminator
    nn.utils.clip_grad_norm_(netD.parameters(), max_grad_norm_d)

    optimizerD.step()

    # Calculate gradient norms
    grad_norm_D = calculate_gradient_norm(netD)

    return lossD.item(), grad_norm_D

def train_generator(netG, netD, criterion, optimizerG, latent_dim, batch_size, device, max_grad_norm_g):
    netG.zero_grad()
    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_images = netG(noise)
    label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    output = netD(fake_images)
    lossG = criterion(output, label)
    lossG.backward()

    if max_grad_norm_g:
        # Gradient clipping for Generator
        nn.utils.clip_grad_norm_(netG.parameters(), max_grad_norm_g)

    optimizerG.step()

    # Calculate gradient norms
    grad_norm_G = calculate_gradient_norm(netG)

    return lossG.item(), grad_norm_G, fake_images

def train_gan():
    # Hyperparameters
    latent_dim = 512
    num_mapping_layers = 10
    img_resolution = 128
    num_synthesis_blocks = int(torch.log2(torch.tensor(img_resolution))) - 2
    lr_d = 1e-5
    lr_g = 1e-4
    beta1 = 0.0
    beta2 = 0.99
    num_epochs = 1000
    batch_size = 50
    max_grad_norm = 16

    # Data loader directory
    data_dir = './flower_data'
    
    # Specify the path to the trained generator model
    trained_generator_path = f'models/generator_epoch_{num_epochs}.pth'
    # Specify the output directory for the generated images
    output_dir = 'generated_images_1024'

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = StyleGANGenerator(latent_dim, num_mapping_layers, num_synthesis_blocks).to(device)
    netD = Discriminator(img_resolution).to(device)

    # Create directories for saving generated images and model weights
    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load the latest models if available
    latest_generator_path, latest_epoch_g = get_latest_model_epoch('models', 'generator')
    latest_discriminator_path, latest_epoch_d = get_latest_model_epoch('models', 'discriminator')

    start_epoch = 0
    if latest_generator_path and latest_discriminator_path:
        map_location = None if device == "cuda" else torch.device('cpu')
        netG.load_state_dict(torch.load(latest_generator_path, map_location=map_location))
        netD.load_state_dict(torch.load(latest_discriminator_path, map_location=map_location))
        start_epoch = max(latest_epoch_g, latest_epoch_d)
        print(f"Loaded latest models from epoch {start_epoch}.")

    # Loss function and optimizers
    criterion = nn.BCEWithLogitsLoss()  # More stable than BCELoss
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Adjust the number of generator updates per discriminator update
        if epoch < num_epochs // 3:
            # Early stages
            num_gen_updates = 1
            max_grad_norm_g = max_grad_norm * 2**1
            max_grad_norm_d = max_grad_norm / 2**1
        elif epoch < 2 * num_epochs // 3:
            # Mid stages
            num_gen_updates = 3
            max_grad_norm_g = max_grad_norm * 2**3
            max_grad_norm_d = max_grad_norm / 2**3
        else:
            # Final stages
            num_gen_updates = 10
            max_grad_norm_g = None
            max_grad_norm_d = max_grad_norm / 2**4

        # Define the transformations for the current epoch
        transform_list = [
            transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
            transforms.Resize((img_resolution, img_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)

        dataset = datasets.Flowers102(root=data_dir, split="train", download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'), 0):
            real_images = data[0].to(device)

            # Update Discriminator network
            lossD, grad_norm_D = train_discriminator(netD, netG, criterion, optimizerD, real_images, latent_dim, device, max_grad_norm_d)

            # Update Generator network multiple times
            for _ in range(num_gen_updates):
                lossG, grad_norm_G, fake_images = train_generator(netG, netD, criterion, optimizerG, latent_dim, batch_size, device, max_grad_norm_g)

            # Report and save fake images
            if i % 5 == 0:
                tqdm.write(f'Step [{i+1}/{len(dataloader)}], Loss D: {lossD:.4f}, Loss G: {lossG:.4f}, Grad Norm D: {grad_norm_D:.4f}, Grad Norm G: {grad_norm_G:.4f}')

                # Save fake images to file
                save_image(fake_images[:25], f'generated_images/fake_images_epoch_{epoch+1}_batch_{i+1}.png', nrow=5, normalize=True)

            if (epoch+1) % 10 == 0:
                # Save the models after each 10 epoch
                torch.save(netG.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
                torch.save(netD.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')

    # Generate enough fake images after training
    generate_fake_images(trained_generator_path, output_dir)

if __name__ == "__main__":
    train_gan()
