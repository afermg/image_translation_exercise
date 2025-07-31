def load_model(pretrained_model_ckpt: str):
    """Load the model from a pretrained checkpoint."""
    # For some reason this imports cellpose, so let's move it inside
    from viscy.light.engine import VSUNet

    phase2fluor_config = dict(
        in_channels=1,
        out_channels=2,
        encoder_blocks=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        decoder_conv_blocks=2,
        stem_kernel_size=(1, 2, 2),
        in_stack_depth=1,
        pretraining=False,
    )
    # Load the model checkpoint
    pretrained_phase2fluor = VSUNet.load_from_checkpoint(
        pretrained_model_ckpt,
        architecture="UNeXt2_2D",
        model_config=phase2fluor_config,
        accelerator="gpu",
    )
    # pretrained_phase2fluor.eval()
    return pretrained_phase2fluor
