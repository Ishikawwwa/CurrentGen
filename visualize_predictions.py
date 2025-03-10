import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# Function to generate a sequence of frames from an input frame
def generate_video(model, input_frame, sequence_length):
    model.eval()
    with torch.no_grad():
        if input_frame.dim() == 2:  # Missing batch and one more dimension
            input_frame = input_frame.unsqueeze(0).unsqueeze(0)
        elif input_frame.dim() == 3:  # Missiong batch
            input_frame = input_frame.unsqueeze(0)
        _, _, h, w = input_frame.size()
        input_frame = input_frame.view(1, -1)
        mean, logvar = model.encode(input_frame)
        z = model.reparameterize(mean, logvar)
        z = z.unsqueeze(1).repeat(1, sequence_length, 1)
        z, _ = model.lstm(z)
        z = z.contiguous().view(sequence_length, -1)
        frames = model.decode(z)
        frames = frames.view(sequence_length, h, w)
    return frames.cpu().numpy()


def visualize_video(frames):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(frames[0], cmap="Blues", vmin=-1, vmax=1)

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    return HTML(ani.to_jshtml())


sample_idx = 0
input_frame = dataset[sample_idx][0]

predicted_frames = generate_video(model, input_frame, sequence_length)
visualize_video(predicted_frames)
