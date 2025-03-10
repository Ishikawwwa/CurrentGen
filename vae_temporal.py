import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


# Main model for generation with temporal features
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        self.lstm = nn.LSTM(latent_dim, latent_dim, num_layers=2, batch_first=True)
        self.sequence_length = sequence_length

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        x = x.view(batch_size * seq_len, -1)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        z = z.view(batch_size, seq_len, -1)
        z, _ = self.lstm(z)
        z = z.contiguous().view(batch_size * seq_len, -1)
        x_recon = self.decode(z)
        x_recon = x_recon.view(batch_size, seq_len, h, w)
        return x_recon, mean, logvar


class OceanCurrentsDataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files) - self.sequence_length + 1

    def __getitem__(self, idx):
        frames = [
            np.load(os.path.join(self.data_dir, self.files[i]))
            for i in range(idx, idx + self.sequence_length)
        ]
        frames = np.stack(frames)
        return torch.tensor(frames, dtype=torch.float32)


def train_vae(model, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    kl_weight = 0.0
    kl_weight_increment = 0.01
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_recon, mean, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(x_recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = (
                recon_loss + kl_weight * kl_loss
            )  # Using dynamic KL weight, because it works slightly better for now
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        scheduler.step()
        kl_weight = min(
            1.0, kl_weight + kl_weight_increment
        )  # Increasing KL weight through time
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, KL Weight: {kl_weight:.2f}"
        )


input_dim = nh * nh  # Grid size for the frames
hidden_dim = 128
latent_dim = 64
sequence_length = 10

model = VAE(input_dim, hidden_dim, latent_dim, sequence_length)
dataset = OceanCurrentsDataset(output_dir, sequence_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

train_vae(model, dataloader, epochs=20, lr=1e-3)
