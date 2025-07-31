import marimo

__generated_with = "0.14.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Part 1
    ## The goal of this section is to train the virtual staining models and inspect their layers.
    - Work through all three parts of the DL@MBL 2024 exercise. You’re welcome to work
    through solution.ipynb. The notebook will walk you through training a 2D image
    translation model that translates the phase channel into fluorescence, and vice versa. It
    also introduces strategies for feature visualization and evaluating the range of
    robustness of the model.
    - Using the phase-> fluorescence and fluorescence-> phase models you have
    trained, visualize the principal components of the feature maps of all stages of the
    encoder and decoder of the model for specific test inputs. This step is also mentioned in
    the notebook: If you are done with the whole checkpoint, you can try to
    look at what your trained model learned.
    - (Bonus task 1) Implement an interactive GUI (using napari or another visualization
    library) that allows you to select an input zarr store, a model checkpoint, and render the
    top three principal components of feature maps at different stages of the encoder and
    decoder. We use OME-zarr format, which you can easily parse using
    iohub.open_ome_zarr method.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import zarr

    from data import get_data
    from models import load_model
    from feature_map import clip_p, composite_nuc_mem, pcs_to_rgb, Color

    # If you are using local files
    # checkpoints_path = Path("./data/06_image_translation/logs/phase2fluor")
    # stores_list = ["./a549_hoechst_cellmask_train_val.ome.zarr", "a549_hoechst_cellmask_test.ome.zarr"]
    checkpoints_path = get_data("training_logs")
    stores_list = [get_data("test"), get_data("train")]
    return (
        Color,
        checkpoints_path,
        clip_p,
        composite_nuc_mem,
        load_model,
        mo,
        np,
        pcs_to_rgb,
        plt,
        stores_list,
        torch,
        zarr,
    )


@app.cell
def _(mo):
    mo.md(r"""## Select checkpoint, evaluation and which image to use as input""")
    return


@app.cell
def _(checkpoints_path, mo, stores_list):
    available_checkpoints = {x.parent.parent.name + "/" + x.name : x for x in checkpoints_path.rglob("*.ckpt")}
    ckpt_selection = mo.ui.dropdown(available_checkpoints.keys(), label="checkpoints", value = list(available_checkpoints.keys())[0])
    data_stores = mo.ui.dropdown([x.name for x in stores_list], label="zarr stores", value = stores_list[0].name)
    stores = {v.name:v for v in stores_list}
    mo.hstack((ckpt_selection,data_stores), justify="start")
    return available_checkpoints, ckpt_selection, data_stores, stores


@app.cell
def _(available_checkpoints, ckpt_selection):
    ckpt_path = available_checkpoints[ckpt_selection.value]
    print(f"Checkpoint selected is located on {ckpt_path}")
    return (ckpt_path,)


@app.cell
def _(data_stores, stores, zarr):
    # for later when we need to use zarr
    with zarr.open(stores[data_stores.value]) as dataset:
        print("Zarr store contents")
        print(dataset.tree())
        # time points
        data_ts = dataset[f"0/0/"]
        fovs = list(data_ts.group_keys())

    return data_ts, fovs


@app.cell
def _(fovs, mo):
    fov = mo.ui.dropdown(fovs, label="FOV to eval", value = sorted(fovs)[0])
    fov
    return (fov,)


@app.cell
def _(data_ts, fov, np):
    img = np.asarray(data_ts[f"{fov.value}/0"])
    orig_shape = img.shape[1:3]
    img = img[:3] # Enforce no z-stack
    phase_img = img[0:1,0:1]
    img.shape
    return img, orig_shape, phase_img


@app.cell
def _(mo, orig_shape):
    nc, nz = orig_shape
    c = mo.ui.slider(0,nc-1, label=f"Channel id (total {nc})", value=0)
    z = mo.ui.slider(0,nz-1, label=f"z stack (total {nz})", value=0, show_value=True)
    mo.vstack((mo.md("Visualise the data (channel doesn't affect which image is used for feature map exploration, but z does)"),
    mo.hstack((c,z), justify="start")))
    return c, z


@app.cell
def _(c, img, plt, z):
    plt.axis("off")
    plt.imshow(img[0,c.value, z.value])
    return


@app.cell
def _(ckpt_path, load_model, np, phase_img, torch, z):
    model = load_model(ckpt_path)
    # Extract features
    with torch.inference_mode():
        # encoder
        encoder_features = model.model.encoder(
            torch.from_numpy(phase_img[:,:,z.value:z.value+1].astype(np.float32)).to(model.device)
        )[0]
        encoder_features_np = [f.detach().cpu().numpy() for f in encoder_features]

        # Print the encoder features shapes
        for f in encoder_features_np:
            print(f.shape)

        # decoder
        features = encoder_features.copy()
        features.reverse()
        current_feat = features[0]
        features.append(None)
        decoder_features_np = []
        for skip, stage in zip(features[1:], model.model.decoder.decoder_stages):
            current_feat = stage(current_feat, skip)
            decoder_features_np.append(current_feat.detach().cpu().numpy())
        for f in decoder_features_np:
            print(f.shape)
        prediction = model.model.head(current_feat).detach().cpu().numpy()
    return decoder_features_np, encoder_features_np, prediction


@app.cell
def _(mo):
    mo.md(r"""## Exploration of feature space""")
    return


@app.cell
def _(
    Color,
    clip_p,
    composite_nuc_mem,
    decoder_features_np,
    encoder_features_np,
    img,
    pcs_to_rgb,
    phase_img,
    plt,
    prediction,
):
    # Defining the colors for plottting the PCA
    BOP_ORANGE = Color(0.972549, 0.6784314, 0.1254902)
    BOP_BLUE = Color(BOP_ORANGE.b, BOP_ORANGE.g, BOP_ORANGE.r)
    GREEN = Color(0.0, 1.0, 0.0)
    MAGENTA = Color(1.0, 0.0, 1.0)

    # Plot the PCA to RGB of the feature maps
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    n_components = 4
    ax[0,0].imshow(phase_img[0, 0, 0], cmap="gray")
    ax[0,0].set_ylabel(phase_img.shape[1:])
    ax[0,0].set_title("Phase")

    ax[1,-1].imshow(clip_p(composite_nuc_mem(img[0,1:3,0], GREEN, MAGENTA)))
    ax[1,-1].set_title("Fluorescence")

    for level, feat in enumerate(encoder_features_np):
        ax[0,level + 1].imshow(pcs_to_rgb(feat, n_components=n_components))
        ax[0,level + 1].set_title(f"Encoder {level + 1}")
        ax[0, level+1].set_ylabel(feat.shape[1:])

    for level, feat in enumerate(decoder_features_np):
        ax[1, level].imshow(pcs_to_rgb(feat, n_components=n_components))
        ax[1, level].set_ylabel(feat.shape[1:])
        ax[1, level].set_title(f"Decoder {level + 1}")

    pred_comp = composite_nuc_mem(prediction[0, :, 0], BOP_BLUE, BOP_ORANGE)
    ax[1,-2].imshow(clip_p(pred_comp))
    ax[1,-2].set_ylabel(prediction.shape[1:])
    ax[1,-2].set_title(f"Prediction")

    for a in ax.ravel():
        a.set(yticklabels=[],yticks=[], xticks=[], xticklabels=[])  

    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
