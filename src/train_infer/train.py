"""
Train a diffusion model on images.
"""

import json, os
import pathlib
from src.utils import dist_util, logger
from src.modeling.diffusion.resample import create_named_schedule_sampler
from script_util import create_model_and_diffusion
from train_loop import TrainLoop
from src.utils import data_utils_sentencepiece
import wandb

from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults

from transformers import set_seed
import os


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()  # DEBUG **
    logger.configure()


    logger.log("creating data loader")

    pathlib.Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    embeddings = data_utils_sentencepiece.create_or_load_embeddings(
        embed_dim=args.in_channel, checkpoint_path=args.checkpoint_path
    )

    train_dataloader = data_utils_sentencepiece.get_dataloader(
        embeddings=embeddings,
        data_path=args.train_txt_path,
        batch_size=args.batch_size,
    )

    val_dataloader = data_utils_sentencepiece.get_dataloader(
        embeddings=embeddings,
        data_path=args.val_txt_path,
        batch_size=args.batch_size,
    )


    args.vocab_size = len(data_utils_sentencepiece.tokenizer)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())  #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"the parameter count is {pytorch_total_params}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"saving the hyperparameters to {args.checkpoint_path}/training_args.json")
    with open(f"{args.checkpoint_path}/training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.debug:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "minimial-text-diffusion"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=val_dataloader,
        eval_interval=args.eval_interval,
    ).run_loop()


if __name__ == "__main__":
    main()
