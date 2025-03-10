import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


# Function to generate frames using the trained model
def generate_frames(model, input_sequence):
    model.eval()
    with torch.no_grad():
        if input_sequence.dim() == 3:  # Missing batch dimension check
            input_sequence = input_sequence.unsqueeze(0)
        batch_size, seq_len, h, w = input_sequence.size()
        input_sequence = input_sequence.view(batch_size * seq_len, -1)
        mean, logvar = model.encode(input_sequence)
        z = model.reparameterize(mean, logvar)
        z = z.view(batch_size, seq_len, -1)
        z, _ = model.lstm(z)
        z = z.contiguous().view(batch_size * seq_len, -1)
        generated_frames = model.decode(z)
        generated_frames = generated_frames.view(batch_size, seq_len, h, w)
    return generated_frames.cpu().numpy()


# Function that is used to compare generated frames as an additional sanity check and debugging step
def visualize_comparison(input_frames, generated_frames):
    num_frames = input_frames.shape[0]
    fig, axes = plt.subplots(2, num_frames, figsize=(15, 4))
    mse_values = []
    for i in range(num_frames):
        print(f"Shape of input frame {i+1}: {input_frames[i].shape}")
        print(f"Shape of generated frame {i+1}: {generated_frames[i].shape}")

        axes[0, i].imshow(input_frames[i], cmap="Blues", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Input Frame {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(generated_frames[i], cmap="Blues", vmin=-1, vmax=1)
        axes[1, i].set_title(f"Generated Frame {i+1}")
        axes[1, i].axis("off")

        mse = np.mean((input_frames[i] - generated_frames[i]) ** 2)
        mse_values.append(mse)
        print(f"Frame {i+1} MSE: {mse:.4f}")
    plt.tight_layout()
    plt.show()
    return mse_values


sample_idx = 0
input_sequence = dataset[sample_idx]


generated_frames = generate_frames(model, input_sequence)

mse_values = visualize_comparison(input_sequence.numpy(), generated_frames)
