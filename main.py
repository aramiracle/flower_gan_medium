import torch
from utils import GANTrainer

torch.cuda.empty_cache()

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 512
    num_mapping_layers = 10
    img_resolution = 128
    lr_d = 1e-5
    lr_g = 1e-4
    beta1 = 0.0
    beta2 = 0.99
    num_epochs = 1000
    batch_size = 50
    max_grad_norm = 16

    gan_trainer = GANTrainer()
    gan_trainer.train(num_epochs=1000)
    gan_trainer.generate_fake_images(generator_path=f'models/generator_epoch_{num_epochs}.pth')
