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
    num_epochs = 400
    batch_size = 48
    max_grad_norm = 32
    data_dir='flower_data'
    models_dir = 'models'
    generated_images_dir='generated_images'

    gan_trainer = GANTrainer(latent_dim, num_mapping_layers, img_resolution, lr_d, lr_g, beta1, beta2, batch_size, max_grad_norm, data_dir, generated_images_dir)
    gan_trainer.train(num_epochs)
    gan_trainer.generate_fake_images(generator_path=f'{models_dir}/generator_epoch_{num_epochs}.pth')
