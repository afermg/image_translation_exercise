"""
- Write a script to infer nuclei and membranes from the phase channel in the above movie.
You’re welcome to use iohub (https://github.com/czbiohub-sf/iohub) for reading the
ome-zarr dataset.
- You’ll notice that the virtually stained nuclei and membrane at the neighboring time
points have intensity fluctuations. Implement a naive baseline of pixel-wise temporal
smoothing to suppress temporal fluctuations.
Extend the inference code with one of your strategies to stabilize the predictions over
time. Compare your strategy with the naive baseline of pixel-wise temporal smoothing of
neighboring timepoints.
- (Bonus task 2) Turn your inference script into a CLI that takes the model checkpoint
and a movie in OME-zarr as input and writes OME-zarr output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from iohub import open_ome_zarr
from models import load_model
from scipy.signal import savgol_filter
from tqdm import tqdm

from data import get_data

pretrained_model_path = get_data("pretrained_ckpt")
movie_path = get_data("movie")

figs_path = Path("./data/figs")
figs_path.mkdir(parents=True, exist_ok=True)


def load_ts(path: Path) -> np.ndarray:
    with open_ome_zarr(
        get_data("movie"),
        mode="r",
        layout="auto",
    ) as dataset:
        dataset.print_tree()
        channel_names = dataset.channel_names
        print(channel_names)
        image_ts = dataset["A/1/1/0"]
        return image_ts


def infer_individual(model, phase_image: np.ndarray):
    # Pad with zeros on the last two dimensions if they do not match the model's size
    padded = np.pad(phase_image, ((0, 0), (0, 0), (0, 0), (0, 2048 - y), (0, 2048 - x)))
    predicted_image_phase2fluor = model(torch.from_numpy(padded).to(model.device))
    return predicted_image_phase2fluor


def calculate_pearson_by_pairs(
    ts: np.ndarray, window_shape: tuple = (2, 300, 300)
) -> np.ndarray:
    "Calculate pearson correlation for contiguous time points."
    window = np.lib.stride_tricks.sliding_window_view(ts, window_shape=window_shape)[
        :, 0, 0, ...
    ]
    # This is not efficient but it is correct so there's that
    coefficients = np.asarray([
        np.corrcoef(x.flatten(), y.flatten())[0, 1] for x, y in window
    ])

    return coefficients


# smooth using a small naive window for contiguous time points
def moving_average(a, n=2, axis=0):
    ret = np.cumsum(a, dtype=float, axis=axis)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# Load image and model
image_ts = load_ts(movie_path)
model = load_model(pretrained_model_path)

# These will come handy later to zoom-in on the results
y, x = image_ts.shape[-2:]
z_stack = image_ts.shape[-3] // 2  # use the stack in the middle of the z-axis

# Let's not make assumptions on the size of the VRAM and use a simple approach
result = []
with torch.inference_mode():  # turn off gradient
    for f, pixels in tqdm(enumerate(image_ts), total=len(image_ts)):
        result.append(
            infer_individual(model, pixels[np.newaxis, :1, z_stack : z_stack + 1])
            .cpu()
            .numpy()
            .squeeze(0)
        )

output = np.asarray(result)

# %% visualise the images
xmin = 200
ymin = 200
tile_size = 300
frame_range = slice(5, 30)
# This can be used for testing
# selected_tps = output[
#     frame_range, :, 0, ymin : ymin + tile_size, xmin : xmin + tile_size
# ]
selected_tps = output[:, :, 0, :]


rolled_avg = np.stack([moving_average(selected_tps[:, i]) for i in range(2)], axis=1)
rolled_avg = np.concatenate((rolled_avg, np.zeros_like(rolled_avg[:1,])), axis=0)

savgol = savgol_filter(selected_tps, axis=0, window_length=10, polyorder=3)

# %% Plot
for i in range(2):
    tiled = np.concatenate(
        np.concatenate(
            [
                x[:, i, ymin : ymin + tile_size, xmin : xmin + tile_size]
                for x in (selected_tps, rolled_avg, savgol)
            ],
            axis=-2,
        ),
        axis=-1,
    )
    # Plot a zoomed-in version
    plt.imshow(tiled)
    plt.ylabel("SG/Window/Raw")
    plt.xlabel("Frames")
    ax = plt.gca()
    ax.set(yticklabels=[], yticks=[], xticks=[], xticklabels=[])
    plt.tight_layout()

    plt.savefig(figs_path / (["Nuclei", "Membrane"][i] + "_combined.png"), dpi=600)
    plt.close()


# %% Validate the correlation of contiguous time points
corrcoefs = {}
for k, v in {"raw": selected_tps, "avg": rolled_avg, "savgol": savgol}.items():
    for i in range(2):
        corrcoefs[(k, ("Nuclei", "Membrane")[i])] = calculate_pearson_by_pairs(v[:, i])

df = pd.DataFrame.from_dict(corrcoefs).reset_index()
df = df.melt(id_vars=[("index", "")])
df.columns = ["Frame", "Smoothing", "Channel", "CorrCoef"]
sns.relplot(
    data=df, x="Frame", y="CorrCoef", col="Channel", hue="Smoothing", kind="line"
)
plt.savefig(figs_path / "corrcoef.png")
plt.close()
