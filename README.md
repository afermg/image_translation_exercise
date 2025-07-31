# Exercise details

## Part 1
 - [x] Notebook visualizing feature maps of virtual staining models for phase -> fluorescence and fluorescence -> phase models.
   - I ran the whole notebook, replacing TODO->DONE where I modified things.
   - I still found it a bit challenging to interpret the feature maps in a significant way, but there may be other different ways to explore that.
 
 - [x] (BONUS) An interactive widget (napari, Jupyter widget, or anything else in Python) for visualizing feature maps for different inputs and different checkpoints.
   - I combined both tasks into the marimo notebook, as it is both a notebook and a widget.
   - Given that I am using a 2D model, I also added a widget to select which z-stack to use, and used both test and train data as inputs.
 
## Part 2
 - [x] A script to improve the temporal regularization of virtually stained structures with an inference-time strategy.
   - For a naive approach I used a smoothing window of length 2, which averages the pixel values over time at the cost of losing one image at the end.
   - My strategy was to use a Savitzky-Golay filter, as they increase precision without distorting the signal (it doesn't try to touch every data point).
   - I finally evaluated all the movies by using pearson correlation between pairs of adjacent time points, which shows that the Savitzky-Golay smooths the pixels over time more than the average window. Whether or not this is fully desirable for every experiment is an independent issue.
 - [X] (BONUS) Inference CLI with naive baseline and your strategy for temporally smooth
 virtual staining.
   - Because I 

## Finishing touches
- [x] Now turn your exercise into a pip-installable package and share installation instructions with README to make it easier for us to reproduce your work.
  - Since I was using `uv` this just required moving the scripts to a folder and including it in `pyproject.toml`
- [x] Ensure the README.md contains all the details requested in each of the tasks and
bonuses.
- [x] Put the outputs of the tasks (Zarr stores) into a Cloud storage service (i.e Google Drive,
iCloud, etc) and share them with us.
- [ ] Please record yourself explaining how you came about each task and bonuses,
problems you encountered while developing this, and anything you would like to
highlight, so we can test it out. (10 minutes or less)

## How to reproduce

### Set up environment
#### Pip
Install a python venv (I built this on python 3.11, so I recommend that one). From within the Python venv install this as the following.
```bash
pip install -e .
```

#### uv
This is the fastest method, install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and run the following line
```bash
uv sync
```

#### Nix
If by some miracle you have a [NixOS](https://nixos.org/) or Linux+Nix installation, you can use this

```bash
nix develop . --impure
```
which should install all dependencies (CUDA included).

### Data download
If you are running things from scratch then you don't need to download any data. The data will be downloaded and cached when you need it.

For the data/models I produced when running the exercise, they are available via [Zenodo](https://zenodo.org/records/16626960) and you can also download them manually. Note that the largest file contains the training logs too, I only have another tarall with those to make it faster to pull it for the interactive notebook.

### Different parts of the exercise
This is how you can run or deploy each notebook/script.

- Original exercise (jupyter notebook): `jupyter lab ./exercise.ipynb`.
- Part 1 (marimo notebook): `marimo edit ./image_translation/part1_inspection_widget.py --no-auth`. I combined both the script and the (bonus) widget using marimo. 
- This file requires some of the training outputs of the first notebook, which you can get from Zenodo automatically, or you can download it manually from here.

For the two notebooks above jupyter and marimo, you may need to do some port-forwarding (`ssh -fNL $PORT:localhost:$PORT $USER@$SERVER), where $PORT is provided by marimo or jupyter, whereas $USER, and $SERVER depend are the info with which you ssh into the machine.

- Part 2 (script): `python ./image_translation/part2_smoothing_script.py`. This will generate a `./data/figs/` folder with a couple of figures that show smoothing via images and correlation plots.
- Part 2 (cli tool): Something like `python python ./image_translation/part2_smoothing_cli.py -c data/06_image_translation/logs/phase2fluor/version_0/checkpoints/epoch=79-step=1680.ckpt -m a549_virtual_staining.ome.zarr/ -o output_dir`

## Notes on changes

### General changes to the original exercise

- Limited torchmetrics upper bound to 1.6, as torchmetric's introduced API changes that break `dice` for VisCy.
- Use [uv](github.com/astral-sh/uv) to be able to fix dependencies, provides reliable reproducibility and it's blazing fast. Additionally, it uses pyproject and follows [PEP 631](https://peps.python.org/pep-0631/).
- Bypassed bash script to pull data and checkpoints, instead I use [pooch](https://github.com/fatiando/pooch?tab=readme-ov-file#example) + a [registry](https://gist.githubusercontent.com/afermg/49bd30211a5c01ac2432edb13943ca3d/raw/fb629fd47040954ea1e9cedd798b9c9ed2ec5669/registry_a549_virtual_staining.txt) to download and cache the zarr files.
- Added dependencies:
- `pooch`: Download and cache files from within Python, it has never given me dependency issues.
- `marimo`: Interactive and reactive notebooks. After one year of usage, it is IMO a superior solution for data analysis and exploration.
- Removed dependencies:
  - `wget`: `wget` is perfect, but it's nice to pull, cache and validate files within python for full reproducibility.
  - `conda`: In general I find it slow, and since most cuda dependencies are pip installable now that it makes sense to use alternative tools for venv management.
- Added support for NixOS, as that is the servers that we use to achieve full reproducibility (it's irrelevant for most people, unless you use/know about Nix).

### Part 1

-   Refactored some visualisation (PCA/plotting) functions into their own script/module
-   Made the plots PCAs of feature maps easier to explore for comparisons
-   Wrapped it all up in a marimo notebook for interactive exploration of both checkpoints and datasets

### Part 2

- I had to pad the last dimensions of the movie, as they did not match the model's shape
- For the last item of part two, I used an average window (which reduced the number of images by one) as a baseline and I aimed to improve on it by using a Savitzky-Golay filter. The upside is that it is more regular, the downside is that it is more blurry.
- I also used Pearson correlation to evaluate whether or not there was a difference
- I use threading to speed up the download of the movie, this may lead to some rare race conditions, but were this to happen it just requires to be run again and it should download things smoothly.

## Maintainance/Development
To make it easier to use scripts that depend on my results I used my `upload_zenodo.sh` and `meta.json` files to upload the tar gz to zenodo. For my own future self, the way to reproduce all results and upload the outputs to zenodo is:

```sh
# Run exercise.ipynb (you could use jupytext or run it from the command line as-is right now)
# It should produce a bunch of files under data/06_image_translation
# For exploration
marimo edit image_translation/part1_inspection_widget.py --no-token
# Save the smoothing figures (not the cli, uses remote files)
python image_translation/part2_smoothing_script.py
# CLI equivalent
#python part2_smoothing_cli.py -c data/06_image_translation/logs/phase2fluor/version_0/checkpoints/epoch=79-step=1680.ckpt -m a549_virtual_staining.ome.zarr/ -o data/smooth_movies`
# Compress outputs
# tar cvf output_training_logs.tar.gz data/06_image_translation/logs/phase2fluor/
# tar cvf image_translation_exercise_outputs.tar.gz data/
# Upload section
# export ZENODO_TOKEN="XXXXXXXXXXXXXXXx"
# bash zenodo_upload
```

