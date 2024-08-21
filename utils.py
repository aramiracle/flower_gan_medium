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
    def __init__(self, latent_dim=512, num_mapping_layers=10, img_resolution=128, lr_d=1e-5, lr_g=1e-4, beta1=0.0, beta2=0.99, batch_size=50, max_grad_norm=16, data_dir='./flower_data', output_dir='generated_images'):
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
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_synthesis_blocks = int(torch.log2(torch.tensor(img_resolution))) - 2

        self.netG = StyleGANGenerator(latent_dim, num_mapping_layers, self.num_synthesis_blocks).to(self.device)
        self.netD = Discriminator(img_resolution).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2))

        self.start_epoch = 0
        self.load_latest_models()

    def load_latest_models(self):
        latest_generator_path, latest_epoch_g = self.get_latest_model_epoch('models', 'generator')
        latest_discriminator_path, latest_epoch_d = self.get_latest_model_epoch('models', 'discriminator')

        if latest_generator_path and latest_discriminator_path:
            map_location = None if self.device == "cuda" else torch.device('cpu')
            self.netG.load_state_dict(torch.load(latest_generator_path, map_location=map_location))
            self.netD.load_state_dict(torch.load(latest_discriminator_path, map_location=map_location))
            self.start_epoch = max(latest_epoch_g, latest_epoch_d)
            print(f"Loaded latest models from epoch {self.start_epoch}.")

    @staticmethod
    def calculate_gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    @staticmethod
    def get_latest_model_epoch(model_dir, model_type):
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type)]
        if not model_files:
            return None, 0

        epochs = [int(re.search(r'\d+', f).group()) for f in model_files]
        latest_epoch = max(epochs)
        latest_model_file = f'{model_type}_epoch_{latest_epoch}.pth'
        return os.path.join(model_dir, latest_model_file), latest_epoch

    def train_discriminator(self, real_images, max_grad_norm_d):
        self.netD.zero_grad()
        b_size = real_images.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)

        output = self.netD(real_images)
        lossD_real = self.criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(b_size, self.latent_dim, device=self.device)
        fake_images = self.netG(noise)
        label.fill_(0)
        output = self.netD(fake_images.detach())
        lossD_fake = self.criterion(output, label)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        nn.utils.clip_grad_norm_(self.netD.parameters(), max_grad_norm_d)
        self.optimizerD.step()

        grad_norm_D = self.calculate_gradient_norm(self.netD)
        return lossD.item(), grad_norm_D

    def train_generator(self, max_grad_norm_g):
        self.netG.zero_grad()
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.netG(noise)
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)
        output = self.netD(fake_images)
        lossG = self.criterion(output, label)
        lossG.backward()

        if max_grad_norm_g:
            nn.utils.clip_grad_norm_(self.netG.parameters(), max_grad_norm_g)

        self.optimizerG.step()

        grad_norm_G = self.calculate_gradient_norm(self.netG)
        return lossG.item(), grad_norm_G, fake_images

    def train(self, num_epochs=1000):
        for epoch in range(self.start_epoch, num_epochs):
            if epoch < num_epochs // 3:
                num_gen_updates = 1
                max_grad_norm_g = self.max_grad_norm * 2**1
                max_grad_norm_d = self.max_grad_norm / 2**1
            elif epoch < 2 * num_epochs // 3:
                num_gen_updates = 3
                max_grad_norm_g = self.max_grad_norm * 2**3
                max_grad_norm_d = self.max_grad_norm / 2**3
            else:
                num_gen_updates = 10
                max_grad_norm_g = None
                max_grad_norm_d = self.max_grad_norm / 2**4

            transform = transforms.Compose([
                transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
                transforms.Resize((self.img_resolution, self.img_resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor()
            ])

            dataset = datasets.Flowers102(root=self.data_dir, split="train", download=True, transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'), 0):
                real_images = data[0].to(self.device)
                lossD, grad_norm_D = self.train_discriminator(real_images, max_grad_norm_d)

                for _ in range(num_gen_updates):
                    lossG, grad_norm_G, fake_images = self.train_generator(max_grad_norm_g)

                if i % 5 == 0:
                    tqdm.write(f'Step [{i+1}/{len(dataloader)}], Loss D: {lossD:.4f}, Loss G: {lossG:.4f}, Grad Norm D: {grad_norm_D:.4f}, Grad Norm G: {grad_norm_G:.4f}')
                    save_image(fake_images[:25], f'{self.output_dir}/fake_images_epoch_{epoch+1}_batch_{i+1}.png', nrow=5, normalize=True)

                if (epoch+1) % 10 == 0:
                    torch.save(self.netG.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
                    torch.save(self.netD.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')

        self.generate_fake_images(f'models/generator_epoch_{num_epochs}.pth')

    def generate_fake_images(self, generator_path, num_images=256, batch_size=64):
        os.makedirs(self.output_dir, exist_ok=True)
        dataset_dir = os.path.join(self.output_dir, 'fake_dataset')
        os.makedirs(dataset_dir, exist_ok=True)

        generator = StyleGANGenerator(self.latent_dim, self.num_mapping_layers, self.num_synthesis_blocks).to(self.device)
        generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        generator.eval()

        num_batches = (num_images + batch_size - 1) // batch_size
        all_images = []

        with torch.no_grad():
            for batch_idx in range(num_batches):
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = generator(noise)

                for i, image in enumerate(fake_images):
                    img_path = os.path.join(dataset_dir, f'fake_image_{batch_idx * batch_size + i + 1}.png')
                    save_image(image, img_path, normalize=True)
                    all_images.append(image)

                print(f'Generated and saved batch {batch_idx + 1}/{num_batches}')

        all_images = torch.stack(all_images)
        grid = make_grid(all_images, nrow=int(torch.sqrt(torch.tensor(num_images))), normalize=True)
        save_image(grid, os.path.join(self.output_dir, 'fake_images_grid.png'), normalize=True)
        print('Saved the image grid as fake_images_grid.png')
