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

class GANTrainer:
    def __init__(self, latent_dim=512, num_mapping_layers=10, img_resolution=128, lr_d=1e-5, lr_g=1e-4, beta1=0.0, beta2=0.99, batch_size=50, max_grad_norm=16, data_dir='./flower_data', models_dir='models', generated_images_dir='generated_images'):
        """
        Initialize the GAN Trainer with necessary parameters and models.

        Args:
            latent_dim (int): Dimensionality of the latent space for the generator.
            num_mapping_layers (int): Number of layers in the mapping network of the StyleGAN generator.
            img_resolution (int): Resolution of the generated images (assumed to be square, e.g., 128x128).
            lr_d (float): Learning rate for the discriminator.
            lr_g (float): Learning rate for the generator.
            beta1 (float): Beta1 hyperparameter for the Adam optimizer.
            beta2 (float): Beta2 hyperparameter for the Adam optimizer.
            batch_size (int): Number of samples per batch.
            max_grad_norm (float): Maximum norm of the gradients for gradient clipping.
            data_dir (str): Directory where the training data is stored.
            models_dir (str): Directory where model checkpoints will be saved.
            output_dir (str): Directory where generated images of each epochs will be saved.

        Sets up the device, initializes the generator and discriminator models, defines the loss function,
        sets up optimizers for both networks, and attempts to load the latest model checkpoints if available.
        """
        self.latent_dim = latent_dim
        self.num_mapping_layers = num_mapping_layers
        self.img_resolution = img_resolution
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.generated_images_dir = generated_images_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate the number of synthesis blocks based on image resolution
        self.num_synthesis_blocks = int(torch.log2(torch.tensor(img_resolution))) - 2

        # Initialize models
        self.netG = StyleGANGenerator(latent_dim, num_mapping_layers, self.num_synthesis_blocks).to(self.device)
        self.netD = Discriminator(img_resolution).to(self.device)

        # Define loss function and optimizers
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))

        # Load the latest model checkpoints if they exist
        self.start_epoch = 0
        self.load_latest_models()

    def load_latest_models(self):
        """
        Load the latest saved models for the generator and discriminator if available.

        This method checks the 'models' directory for the latest checkpoints for both
        the generator and discriminator. If found, the model states are loaded from these
        checkpoints, and training resumes from the saved epoch.

        Sets:
            start_epoch (int): The epoch number to resume training from after loading the latest models.
        """
        # Check if the models directory exists
        if not os.path.exists(self.models_dir):
            print(f"Model directory '{self.models_dir}' does not exist. Starting training from scratch...")
            self.start_epoch = 0
            return

        try:
            latest_generator_path, latest_epoch_g = self.get_latest_model_epoch(self.models_dir, 'generator')
            latest_discriminator_path, latest_epoch_d = self.get_latest_model_epoch(self.models_dir, 'discriminator')

            if latest_generator_path and latest_discriminator_path:
                map_location = None if self.device == "cuda" else torch.device('cpu')
                self.netG.load_state_dict(torch.load(latest_generator_path, map_location=map_location))
                self.netD.load_state_dict(torch.load(latest_discriminator_path, map_location=map_location))
                self.start_epoch = max(latest_epoch_g, latest_epoch_d)
                print(f"Loaded latest models from epoch {self.start_epoch}.")
            else:
                print("No previous checkpoints found in the directory. Starting training from scratch...")

        except Exception as e:
            print(f"An error occurred while loading models: {e}")
            self.start_epoch = 0


    @staticmethod
    def calculate_gradient_norm(model):
        """
        Calculate the gradient norm of the model parameters.

        Args:
            model (torch.nn.Module): The model for which to calculate the gradient norm.

        Returns:
            float: The L2 norm of the gradients of the model parameters.

        This method iterates over all the model's parameters, computes the L2 norm of the gradients
        for each parameter, and returns the total L2 norm.
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    @staticmethod
    def get_latest_model_epoch(model_dir, model_type):
        """
        Get the path and epoch number of the latest saved model for the specified type.

        Args:
            model_dir (str): Directory where model checkpoints are stored.
            model_type (str): Type of model ('generator' or 'discriminator').

        Returns:
            tuple: (str, int) Path to the latest model checkpoint and its epoch number. Returns (None, 0) if no models are found.

        This method lists all files in the model directory, extracts the epoch numbers from the filenames,
        and determines the file corresponding to the latest epoch.
        """
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type)]
        if not model_files:
            return None, 0

        epochs = [int(re.search(r'\d+', f).group()) for f in model_files]
        latest_epoch = max(epochs)
        latest_model_file = f'{model_type}_epoch_{latest_epoch}.pth'
        return os.path.join(model_dir, latest_model_file), latest_epoch

    def train_discriminator(self, real_images, max_grad_norm_d):
        """
        Train the discriminator network with both real and fake images for one step.

        Args:
            real_images (torch.Tensor): A batch of real images from the dataset.
            max_grad_norm_d (float): Maximum gradient norm for discriminator gradient clipping.

        Returns:
            tuple: (float, float) Loss value for the discriminator and its gradient norm.

        This method performs one step of training for the discriminator. It calculates the loss for both
        real and fake images, backpropagates the loss, clips the gradients, and updates the discriminator's weights.
        """
        self.netD.zero_grad()
        b_size = real_images.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)

        # Forward pass with real images
        output = self.netD(real_images)
        lossD_real = self.criterion(output, label)
        lossD_real.backward()

        # Generate fake images and forward pass with them
        noise = torch.randn(b_size, self.latent_dim, device=self.device)
        fake_images = self.netG(noise)
        label.fill_(0)
        output = self.netD(fake_images.detach())
        lossD_fake = self.criterion(output, label)
        lossD_fake.backward()

        # Compute total loss and update Discriminator
        lossD = lossD_real + lossD_fake
        if max_grad_norm_d:
            # Clip gradients if max gradient norm is set
            nn.utils.clip_grad_norm_(self.netD.parameters(), max_grad_norm_d)
        self.optimizerD.step()

        # Compute gradient norm for the Discriminator
        grad_norm_D = self.calculate_gradient_norm(self.netD)
        return lossD.item(), grad_norm_D

    def train_generator(self, max_grad_norm_g):
        """
        Train the generator network to produce more realistic images for one step.

        Args:
            max_grad_norm_g (float or None): Maximum gradient norm for generator gradient clipping. If None, no clipping is applied.

        Returns:
            tuple: (float, float, torch.Tensor) Loss value for the generator, its gradient norm, and the generated fake images.

        This method performs one step of training for the generator. It generates fake images, computes
        the loss based on how well they fool the discriminator, backpropagates the loss, optionally clips gradients,
        and updates the generator's weights.
        """
        self.netG.zero_grad()
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.netG(noise)
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)
        output = self.netD(fake_images)
        lossG = self.criterion(output, label)
        lossG.backward()

        if max_grad_norm_g:
            # Clip gradients if max gradient norm is set
            nn.utils.clip_grad_norm_(self.netG.parameters(), max_grad_norm_g)

        self.optimizerG.step()

        # Compute gradient norm for the Generator
        grad_norm_G = self.calculate_gradient_norm(self.netG)
        return lossG.item(), grad_norm_G, fake_images

    def train(self, num_epochs=1000):
        """
        Main training loop for the GAN. Updates the Discriminator and Generator over multiple epochs.

        Args:
            num_epochs (int): Number of epochs to train the GAN.

        This method handles the overall training process, including:
        - Loading the dataset and setting up the data loader.
        - Training both the discriminator and generator for the specified number of epochs.
        - Dynamically adjusting the number of generator updates and gradient clipping norms based on the epoch.
        - Saving intermediate results and model checkpoints periodically.
        """
        # Create directories for saving generated images and model weights
        os.makedirs(self.generated_images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        for epoch in range(self.start_epoch, num_epochs):
            # Adjust training parameters based on epoch
            if epoch < num_epochs // 2:
                stage = 1
                num_gen_updates = 1
                max_grad_norm_g = self.max_grad_norm * 2**1
                max_grad_norm_d = self.max_grad_norm / 2**1
            elif epoch < 3 * num_epochs // 4:
                stage = 2
                num_gen_updates = 3
                max_grad_norm_g = self.max_grad_norm * 2**3
                max_grad_norm_d = self.max_grad_norm / 2**3
            else:
                stage = 3
                num_gen_updates = 5
                max_grad_norm_g = None
                max_grad_norm_d = None

            # Define transformations for image preprocessing
            transform = transforms.Compose([
                transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
                transforms.Resize((self.img_resolution, self.img_resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor()
            ])

            # Load dataset and dataloader
            dataset = datasets.Flowers102(root=self.data_dir, split="train", download=True, transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'), 0):
                real_images = data[0].to(self.device)
                lossD, grad_norm_D = self.train_discriminator(real_images, max_grad_norm_d)

                # Update the Generator multiple times
                for _ in range(num_gen_updates):
                    lossG, grad_norm_G, fake_images = self.train_generator(max_grad_norm_g)

                # Report progress and save images every few steps
                if i % 5 == 0:
                    tqdm.write(f'Stage [{stage}/3], Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss D: {lossD:.4f}, Loss G: {lossG:.4f}, Grad Norm D: {grad_norm_D:.4f}, Grad Norm G: {grad_norm_G:.4f}')
                    save_image(fake_images[:25], f'{self.generated_images_dir}/fake_images_epoch_{epoch+1}_batch_{i+1}.png', nrow=5, normalize=True)

            # Save models every 10 epochs
            if (epoch+1) % 10 == 0:
                torch.save(self.netG.state_dict(), f'{self.models_dir}/generator_epoch_{epoch+1}.pth')
                torch.save(self.netD.state_dict(), f'{self.models_dir}/discriminator_epoch_{epoch+1}.pth')

    def generate_fake_images(self, generator_path, num_images=256, batch_size=64):
        """
        Generate and save fake images using the trained generator.

        Args:
            generator_path (str): Path to the trained generator model checkpoint.
            num_images (int): Total number of fake images to generate.
            batch_size (int): Number of images to generate per batch.

        This method loads a trained generator, uses it to generate a specified number of fake images,
        saves each generated image individually, and also creates a grid of all generated images.

        The generated images are saved in the output directory, and a grid of images is saved as 'fake_images_grid.png'.
        """
        # Create output directories if they don't exist
        os.makedirs(self.generated_images_dir, exist_ok=True)
        dataset_dir = os.path.join(self.generated_images_dir, 'fake_dataset')
        os.makedirs(dataset_dir, exist_ok=True)

        # Load the trained generator model
        generator = StyleGANGenerator(self.latent_dim, self.num_mapping_layers, self.num_synthesis_blocks).to(self.device)
        generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        generator.eval()

        num_batches = (num_images + batch_size - 1) // batch_size
        all_images = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = generator(noise)

                # Save each fake image
                for i, image in enumerate(fake_images):
                    img_path = os.path.join(dataset_dir, f'fake_image_{batch_idx * batch_size + i + 1}.png')
                    save_image(image, img_path, normalize=True)
                    all_images.append(image)

                print(f'Generated and saved batch {batch_idx + 1}/{num_batches}')

        # Save a grid of all generated images
        all_images = torch.stack(all_images)
        grid = make_grid(all_images, nrow=int(torch.sqrt(torch.tensor(num_images))), normalize=True)
        save_image(grid, os.path.join(self.generated_images_dir, 'fake_images_grid.png'), normalize=True)
        print('Saved the image grid as fake_images_grid.png')
