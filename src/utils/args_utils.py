"""
Utilities for command line arguments.
"""

import argparse



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
        save_interval=25000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path="diff_models",
        train_txt_path="data/quotes_train.txt",
        val_txt_path="data/quotes_valid.txt",
        dataset="",
        notes="",
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
        tok_thresh=150
    )
    
    guided_generation_defaults = dict(
        classifier_num_epochs=15
    )
    
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    defaults.update(guided_generation_defaults)
    defaults.update(decoding_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    add_dict_to_argparser(parser, defaults)
    return parser


def model_and_diffusion_defaults():
    """
    Defaults for text-diffusion model training.
    """
    return dict(
        sequence_len=64,
        num_channels=16,
        num_heads=4,
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
        model_arch="transformer",
        in_channel=16,
        out_channel=16,
        vocab_size=66,
        config_name="bert-base-uncased",
        logits_mode=1,
        training_mode="diffusion-lm",
        init_pretrained=False,
        freeze_embeddings=False,
        use_pretrained_embeddings=True,
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
        classifier_path="",
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
