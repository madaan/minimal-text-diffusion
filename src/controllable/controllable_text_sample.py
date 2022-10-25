"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os, json
import sys
from typing import List
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from functools import partial
from src.utils import dist_util, logger


from src.utils.args_utils import *
from train_infer.factory_methods import create_model_and_diffusion
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.utils.custom_tokenizer import create_tokenizer
from src.controllable.langevin import langevin_binary_classifier
from src.controllable.classifier import DiffusionBertForSequenceClassification


def main():

    args = create_argparser().parse_args()

    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]

    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    # overwrite this because we want to allow generation for any diffusion step.
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args["model_name_or_path"] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args["out_dir"] = args.out_dir
    training_args["num_samples"] = args.num_samples

    args.__dict__.update(training_args)
    args.sigma_small = True

    logger.info(f"Init pretrained = {args.init_pretrained}")
    logger.info(f"Freeze embeddings = {args.freeze_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_embeddings}")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    tokenizer = create_tokenizer(
        return_pretokenized=args.use_pretrained_embeddings, path=f"data/{args.dataset}/"
    )

    model.config.update({"embedding_dim": args.in_channel})
    model.config.update({"train_diffusion_steps": args.diffusion_steps})
    model.config.update({"vocab_size": tokenizer.vocab_size})

    classifier = DiffusionBertForSequenceClassification.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path + "/classifier.pt",
        config=model.config,
        diffusion_model=diffusion,
    ).to("cuda")

    # freeze the classifier
    for param in classifier.parameters():
        param.requires_grad = False

    langevin_classifier_wrapper = partial(langevin_binary_classifier, classifier=classifier)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()  # DEBUG

    logger.log(f"Generating {args.num_samples} samples")
    logger.log(f"Clamping is set to {args.clamp}")
    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_shape = (args.batch_size, args.sequence_len, model.word_embedding.weight.shape[1])
        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            progress=True,
            tokenizer=tokenizer,
            log_verbose=True,
            langevin_fn=langevin_classifier_wrapper,
        )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

    logger.log(f"created {len(all_samples)} samples")

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples * args.mbr_sample]

    x_t = th.tensor(arr).cuda()

    logits = model.get_logits(x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)

    decoded_sentences = []

    for seq in cands.indices:
        decoded_sentence = tokenizer.decode(seq.squeeze(1).tolist())
        decoded_sentences.append(decoded_sentence)

    dist.barrier()
    logger.log("sampling complete")

    write_outputs(args=args, sentences=decoded_sentences)


def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(tokenizer.vocab_size, emb_dim)
    embeddings.load_state_dict(th.load(f"{checkpoint_path}/random_emb.torch"))
    return embeddings


def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def write_outputs(args: dict, sentences: List[str]) -> None:

    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]

    num_samples = len(sentences)
    output_file_basepath = (
        os.path.join(
            model_dir,
            f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}",
        )
        + ".txt.ctrl"
    )

    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence in sentences:
            text_fout.write(generated_sentence + "\n")

        print(f"written the decoded output to {output_file_basepath}")


if __name__ == "__main__":
    main()
