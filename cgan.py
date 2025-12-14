"""
Conditional GAN (Pix2Pix-style) for image denoising.
Uses adversarial loss + L1 reconstruction loss.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Dataset
transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, noisy_img, clean_img):
        x = torch.cat([noisy_img, clean_img], dim=1)
        return self.model(x)


# Noise function
def add_noise(images, noise_factor=0.3):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0., 1.)


# Models
G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
l1_loss = nn.L1Loss()
lambda_l1 = 150

optimizer_G = optim.Adam(G.parameters(), lr=0.001)
optimizer_D = optim.Adam(D.parameters(), lr=0.001)

real_label = 1.
fake_label = 0.


# Training
num_epochs = 5

for epoch in range(num_epochs):
    for imgs, _ in dataloader:

        clean_imgs = imgs.view(imgs.size(0), -1)
        noisy_imgs = add_noise(clean_imgs)
        batch_size = clean_imgs.size(0)

        # Train Discriminator
        D.zero_grad()

        real_targets = torch.full((batch_size, 1), real_label)
        fake_targets = torch.full((batch_size, 1), fake_label)

        real_loss = criterion(D(noisy_imgs, clean_imgs), real_targets)

        generated_imgs = G(noisy_imgs).detach()
        fake_loss = criterion(D(noisy_imgs, generated_imgs), fake_targets)

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        optimizer_D.step()

        # Train Generator
        G.zero_grad()

        generated_imgs = G(noisy_imgs)

        gan_loss = criterion(D(noisy_imgs, generated_imgs), real_targets)
        recon_loss = l1_loss(generated_imgs, clean_imgs)

        gen_loss = gan_loss + lambda_l1 * recon_loss
        gen_loss.backward()
        optimizer_G.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"D Loss: {disc_loss.item():.4f} | G Loss: {gen_loss.item():.4f}"
    )


# Evaluation
G.eval()
with torch.no_grad():
    img, _ = dataset[0]
    clean_img = img.view(1, -1)
    noisy_img = add_noise(clean_img)
    generated_img = G(noisy_img)


# Visualization
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(clean_img.view(28, 28), cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_img.view(28, 28), cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(generated_img.view(28, 28), cmap="gray")
plt.title("cGAN Output")
plt.axis("off")

plt.tight_layout()
plt.show()


# Save output
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "/Users/grvchanr/Code/image_restoration/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.savefig(os.path.join(RESULTS_DIR, "cgan_l1.png"),
            bbox_inches="tight")
