import os
import fluiddyn as fld
from fluidsim.solvers.ns2d.solver import Simul
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import Video


t_end = 10.0  # Total simulation time
nh = 64  # Grid size
Lh = 10.0  # Domain size

params = Simul.create_default_params()

params.short_name_type_run = "ocean_currents"
params.oper.nx = params.oper.ny = nh
params.oper.Lx = params.oper.Ly = Lh

params.nu_8 = 1e-4  # Reduced viscosity for smoother motion, do not change much
params.forcing.forcing_rate = 0.1
params.time_stepping.t_end = t_end
params.init_fields.type = "dipole"  # Initial condition to start the system
params.forcing.enable = True
params.forcing.type = "tcrandom"  # Turbulent forcing

params.output.ONLINE_PLOT_OK = False
params.output.periods_plot.phys_fields = 0.0
params.output.periods_save.phys_fields = 0.0

sim = Simul(params)

output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)

frames = []
current_time = 0.0
dt_frame = 0.1


def create_frame():
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, Lh)
    ax.set_ylim(0, Lh)
    ax.set_aspect("equal")
    ax.set_facecolor("midnightblue")
    return fig, ax


fig, ax = create_frame()
im = ax.imshow(np.zeros((nh, nh)), cmap="Blues", extent=[0, Lh, 0, Lh], vmin=-1, vmax=1)
plt.colorbar(im, label="Vorticity")
title = ax.set_title("Ocean Currents (Time: 0.0s)")

# Function to update the animation
frame_count = 0


def update(frame):
    global current_time, frame_count
    while sim.time_stepping.t < current_time:
        sim.time_stepping.one_time_step()
    current_time += dt_frame

    vorticity = sim.state.get_var("rot")
    im.set_array(vorticity)
    title.set_text(f"Ocean Currents (Time: {current_time:.1f}s)")

    # Save vorticity field
    np.save(os.path.join(output_dir, f"frame_{frame_count:04d}.npy"), vorticity)
    frame_count += 1
    return im, title


ani = FuncAnimation(fig, update, frames=int(t_end / dt_frame), interval=100, blit=True)

output_path = "ocean_currents.mp4"
writer = FFMpegWriter(fps=10, metadata=dict(artist="You"), bitrate=1800)
ani.save(output_path, writer=writer)

Video(output_path, embed=True)
