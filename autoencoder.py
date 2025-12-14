"""
Denoising Autoencoder for image restoration.
Uses L1 loss to reconstruct clean images from noisy inputs.
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

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)


# Autoencoder model
class AutoEncoder(nn.Module):
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder()

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Noise function
def add_noise(images, noise_factor=0.3):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0., 1.)


# Training
num_epochs = 5

for epoch in range(num_epochs):
    for imgs, _ in dataloader:

        clean_imgs = imgs.view(imgs.size(0), -1)
        noisy_imgs = add_noise(clean_imgs)

        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluation
model.eval()
with torch.no_grad():
    img, _ = dataset[0]
    clean_img = img.view(1, -1)
    noisy_img = add_noise(clean_img)
    reconstructed_img = model(noisy_img)


# Reshape for visualization
clean_img = clean_img.view(28, 28)
noisy_img = noisy_img.view(28, 28)
reconstructed_img = reconstructed_img.view(28, 28)


# Plot results
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(clean_img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_img, cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.show()


# Save output
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "/Users/grvchanr/Code/image_restoration/results")

os.makedirs(RESULTS_DIR, exist_ok=True)

plt.savefig(os.path.join(RESULTS_DIR, "autoencoder_l1.png"),
            bbox_inches="tight")
