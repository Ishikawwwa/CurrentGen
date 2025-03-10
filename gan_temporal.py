import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


# Currently still working on this
# GAN approach
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim, sequence_length):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers=2, batch_first=True)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Tanh())
        self.sequence_length = sequence_length

    def forward(self, z):
        z, _ = self.lstm(z)
        z = z.permute(0, 2, 1)
        z = self.conv_layers(z)
        z = z.permute(0, 2, 1)
        z = z.contiguous().view(-1, z.size(-1))
        output = self.fc(z)
        output = output.view(-1, self.sequence_length, output.size(-1))
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, x.size(-1))
        output = self.fc(x)
        return output.view(-1)


def train_gan(generator, discriminator, dataloader, epochs=10, lr=1e-4):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
    for epoch in range(epochs):
        for real_sequences in dataloader:
            batch_size = real_sequences.size(0)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)

            optimizer_d.zero_grad()
            outputs = discriminator(real_sequences)
            real_loss = criterion(outputs, real_labels)
            real_loss.backward()

            noise = torch.randn(batch_size, generator.sequence_length, noise_dim)
            fake_sequences = generator(noise)
            outputs = discriminator(fake_sequences.detach())
            fake_loss = criterion(outputs, fake_labels)
            fake_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            outputs = discriminator(fake_sequences)
            generator_loss = criterion(outputs, real_labels)
            generator_loss.backward()
            optimizer_g.step()

        print(
            f"Epoch {epoch + 1}, D Loss: {real_loss.item() + fake_loss.item()}, G Loss: {generator_loss.item()}"
        )


noise_dim = 100
hidden_dim = 128
output_dim = nh * nh
sequence_length = 10

generator = Generator(noise_dim, hidden_dim, output_dim, sequence_length)
discriminator = Discriminator(output_dim, hidden_dim, sequence_length)
dataset = OceanCurrentsDataset(output_dir, sequence_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

train_gan(generator, discriminator, dataloader, epochs=20, lr=1e-4)
