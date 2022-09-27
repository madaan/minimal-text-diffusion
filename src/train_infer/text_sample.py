"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os, json
from typing import List

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from src.modeling.diffusion.rounding import load_embeddings_and_tokenizer
from transformers import AutoTokenizer

from src.utils.test_util import get_weights, denoised_fn_round

from src.utils import dist_util, logger
from functools import partial

from src.utils.args_utils import *
from src.train_infer.script_util import (
    create_model_and_diffusion,
)
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults


# BAD: this should not be global
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def main():
    set_seed(101)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]

    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args['model_name_or_path'] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args['out_dir'] = args.out_dir

    args.__dict__.update(training_args)
    args.sigma_small = True

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()  # DEBUG

    embeddings = load_embeddings(
        checkpoint_path=args.checkpoint_path, emb_dim=args.in_channel, tokenizer=tokenizer
    )
    embeddings.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    logger.log("sampling...")
    logger.log(f"Clamping is set to {args.clamp}")
    all_samples = []
    embeddings_with_grad_false = get_weights(embeddings, args)
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_shape = (args.batch_size, args.sequence_len, args.in_channel)
        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, embeddings_with_grad_false.cuda())
            if args.clamp == "clamp"
            else None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            progress=True,
            tokenizer=tokenizer,
        )


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples * args.mbr_sample]

    x_t = th.tensor(arr).cuda()

    logits = model.get_logits(x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)

    decoded_sentences = []

    for seq in cands.indices:
        decoded_sentence = tokenizer.decode(seq.squeeze(1), clean_up_tokenization_spaces=True, skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)

    dist.barrier()
    logger.log("sampling complete")

    write_outputs(args=args, sentences=decoded_sentences)


def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(len(tokenizer), emb_dim)
    embeddings.load_state_dict(th.load(f'{checkpoint_path}/random_emb.torch'))
    return embeddings


def read_training_args(config_path):
    with open(
        config_path,
        "rb",
    ) as f:
        return json.load(f)


def write_outputs(args: dict, sentences: List[str]) -> None:

    model_base_name = (
        os.path.basename(os.path.split(args.model_name_or_path)[0])
        + f".{os.path.split(args.model_name_or_path)[1]}"
    )
    output_file_basepath = os.path.join(
        args.out_dir, f"{model_base_name}.samples_{args.top_p}.steps={args.diffusion_steps}.clamp={args.clamp}"
    )

    with open(output_file_basepath + ".txt", "w") as text_fout, open(
        output_file_basepath + ".json", "w"
    ) as json_fout:
        for generated_sentence in sentences:
            text_fout.write(generated_sentence + "\n")
            json_fout.write(json.dumps([generated_sentence]) + "\n")

        print(f"written the decoded output to {output_file_basepath}")

    print(sentences[:2])

    with open("generation_outputs_emb128/two_steps_sanity_check.txt", "r") as fin:
        sanity_check_lines = fin.readlines()

    # compare with sentences
    for i, sanity_check_line in enumerate(sanity_check_lines):
        sanity_check_line_toks = set(sanity_check_line.strip().split())
        generated_sentence_toks = set(sentences[i].split())

        common = sanity_check_line_toks.intersection(generated_sentence_toks)
        jaccard = len(common) / len(sanity_check_line_toks.union(generated_sentence_toks))
        # assert (
        #     jaccard > 0.9
        # ), f"line {i} is not similar enough: {jaccard} {sanity_check_line_toks} {generated_sentence_toks} | {common}"


if __name__ == "__main__":
    main()
