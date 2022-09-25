"""
Utilities for command line arguments.
"""

import argparse

from modeling.diffusion.nn import checkpoint


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=30000,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path="diff_models",
        train_txt_path="data/quotes_train.txt",
        val_txt_path="data/quotes_valid.txt",
    )
    text_defaults = dict(
        modality="text",
        emb_scale_factor=1.0,
        in_channel=16,
        out_channel=16,
        noise_level=0.0,
        cache_mode="no",
        use_bert_tokenizer="no",
        padding_mode="block",
        preprocessing_num_workers=1,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    defaults.update(decoding_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test_run", action="store_true")

    add_dict_to_argparser(parser, defaults)
    return parser


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=16,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=10000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        model_arch="trans-unet",
        in_channel=16,
        out_channel=16,
        training_mode="e2e",
        vocab_size=66,
        config_name="bert-base-uncased",
        experiment_mode="lm",
        logits_mode=1,
    )


def decoding_defaults():
    return dict(
        num_samples=50,
        top_p=0.9,
        out_dir="",
        model_name_or_path="",
        checkpoint_path="",
        use_ddim=False,
        clip_denoised=False,
        batch_size=64,
        mbr_sample=1,
        verbose="yes",
        clamp="clamp",
        preprocessing_num_workers=1,
        emb_scale_factor=1.0,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
