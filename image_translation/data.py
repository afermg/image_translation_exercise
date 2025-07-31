#!/usr/env/jupyter
"""Explicitly define the input data in a function. I do some threading to maximise download speed."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PosixPath

from pooch import Unzip, create, os_cache, retrieve, Untar


def get_data(name: str, out_dir: str = None) -> PosixPath:
    # I replaced the original zarr files with zipped omezarrs to be able to hash using pooch.
    if name == "pretrained_ckpt":
        path = retrieve(
            "https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt",
            known_hash="1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe",
        )

    elif name == "train":
        path = retrieve(
            "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/training/a549_hoechst_cellmask_train_val.ome.zarr.zip",
            known_hash="68d741b5a3be29f45d0acb5f96d8717657f6d7292bbd1015e474a4bdd9b7ec47",
            processor=Unzip(),
        )

    elif name == "test":
        path = retrieve(
            "https://public.czbiohub.org/comp.micro/viscy/VS_datasets/VSCyto2D/test/a549_hoechst_cellmask_test.ome.zarr.zip",
            known_hash="ae16ff735a25179e81fd7df5a762de25e5cb9d633568f001d4f425657aa3bfa9",
            processor=Unzip(),
        )
    elif name == "movie":  # Since this is a zarr it requires a bit more work
        pooch_entry = create(
            path=os_cache("translation_zarr"),
            base_url="https://public.czbiohub.org/royerlab/ultrack/a549_virtual_staining.ome.zarr",
        )
        pooch_entry.load_registry(
            # I put the pooch registry in a Github Gist for easy access.
            fname=retrieve(
                "https://gist.githubusercontent.com/afermg/49bd30211a5c01ac2432edb13943ca3d/raw/fb629fd47040954ea1e9cedd798b9c9ed2ec5669/registry_a549_virtual_staining.txt",
                known_hash="61a697478c7187c04ff89037c09264a6bbece80aa0e5917d68671c992429ce72",
            )
        )
        # Download the files if unavailable
        with ThreadPoolExecutor() as ex:
            _ = list(
                ex.map(
                    pooch_entry.fetch,
                    pooch_entry.registry.keys(),
                )
            )
        path = pooch_entry.abspath

    elif (
        name == "training_logs"
    ):  # I added this after finishing some rounds of training, so as to be able to run the widgets consistently
        path = retrieve(
            "https://zenodo.org/api/records/16626960/files/output_training_logs.tar.gz/content",
            path=os_cache("training_logs"),
            known_hash="44457235fafcbbe759c66a04e8443219b824cf21f7f433e2283bc3cc3013052a",
            processor=Untar(),
        )
        # Pointing to phase2fluor/ directory
        path = Path(path[0]).parent.parent
    if isinstance(path, list):
        path = path[0]
        path = Path(path).parent
    path = Path(path)
    return path
