"""
Run this as as `python part2_smoothing_cli.py -c data/06_image_translation/logs/phase2fluor/version_0/checkpoints/epoch=79-step=1680.ckpt -m a549_virtual_staining.ome.zarr/ -o output_dir`

- (Bonus task 2) Turn your inference script into a CLI that takes the model checkpoint
and a movie in OME-zarr as input and writes OME-zarr output.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from iohub import open_ome_zarr
from scipy.signal import savgol_filter
from tqdm import tqdm

from models import load_model


def load_ts(path: Path) -> np.ndarray:
    with open_ome_zarr(
        path,
        mode="r",
        layout="auto",
    ) as dataset:
        dataset.print_tree()
        channel_names = dataset.channel_names
        print(channel_names)
        # Note that this only works in this case, further engineering would be required to generalise
        image_ts = dataset["A/1/1/0"]
        return image_ts


def infer_individual(model, phase_image: np.ndarray):
    y, x = phase_image.shape[-2:]

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


def main(
    args,
):
    # parsed args
    pretrained_model_path = args.checkpoint
    movie_path = args.movie
    stores_path = Path(args.output)

    # Validation
    assert str(stores_path), "Path for stores is empty"

    # Default args
    savgol_window_length = 10
    savgol_polyorder = 3

    ## Assignment ends

    # Load image and model
    image_ts = load_ts(movie_path)
    model = load_model(pretrained_model_path)

    # These will come handy later to zoom-in on the results
    y, x = image_ts.shape[-2:]
    z_stack = image_ts.shape[-3] // 2  # use the stack in the middle of the z-axis

    # Let's not make assumptions on the size of the VRAM and use a simple approach
    result = []
    print("Applying pretrained model to movie")
    with torch.inference_mode():  # turn off gradient
        for f, pixels in tqdm(enumerate(image_ts), total=len(image_ts)):
            result.append(
                infer_individual(model, pixels[np.newaxis, :1, z_stack : z_stack + 1])
                .cpu()
                .numpy()
                .squeeze(0)
            )

    virtual_stains = np.asarray(result)[:, :, 0]

    rolled_avg = np.stack(
        [moving_average(virtual_stains[:, i]) for i in range(2)], axis=1
    )
    rolled_avg = np.concatenate((rolled_avg, np.zeros_like(rolled_avg[:1,])), axis=0)

    savgol = savgol_filter(
        virtual_stains,
        axis=0,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )
    smooth_data = {"window": rolled_avg, "savgol": savgol}

    # %% Write
    print(f"Saving time-aware inference into {stores_path}")
    for name, data in smooth_data.items():
        print(f"Writing smooth predictions based on {name} method")
        with_z = data[:, :, np.newaxis]
        shape = with_z.shape
        with open_ome_zarr(
            stores_path / name,
            layout="fov",
            mode="w",
            channel_names=["Nuclei", "Membrane"],
        ) as dataset:
            img = dataset.create_zeros(
                name="0",
                shape=shape,
                dtype=np.float64,
                chunks=(1, 1, 1, *shape[-2:]),  # chunk by XY planes
            )
            for t, snapshot in tqdm(enumerate(with_z)):
                # write 4D image data for the time point
                img[t] = snapshot
            dataset.print_tree()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Give me a model and movie to get the smoothed virtual staining over time."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Location of the checkpount.",
    )
    parser.add_argument(
        "-m", "--movie", type=str, required=True, help="Location of the ome_zarr movie."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output directory."
    )

    args = parser.parse_args()

    ## Parsing ends
    main(args)
