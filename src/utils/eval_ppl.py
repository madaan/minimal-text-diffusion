"""
Evaluates perplexity of a model on a dataset.
Directly taken from https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py
"""

from itertools import chain
from typing import List
import torch

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)


def compute_perplexity(data, batch_size: int = 16, add_start_token: bool = True):



    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = model.config.max_length - 1
    else:
        max_tokenized_len = model.config.max_length

    encodings = tokenizer(
        data,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_perplexity_for_file(path: str):
    # read lines
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>"]
    with open(path, "r") as f:
        lines = f.readlines()
    
    lines = [remove_all(line.strip(), special_tokens) for line in lines if len(line.strip()) > 0]
    try:
        num_unique_lines = len(set(lines))
        perc_unique_lines = round(num_unique_lines * 100 / len(lines), 2)
        all_tokens = list(chain(*[line.split() for line in lines]))
        perc_unique_tokens = round(len(set(all_tokens)) * 100 / len(all_tokens), 2)
        
        return {"data": lines, "ppl": compute_perplexity(lines)['mean_perplexity'], "perc_unique_lines": perc_unique_lines, "perc_unique_tokens": perc_unique_tokens}
    except Exception as e:
        return {"data": [], "ppl": 1e6, "perc_unique_lines": 0, "perc_unique_tokens": 0}
        

def remove_all(line: str, special_toks: List[str]) -> str:
    for tok in special_toks:
        line = line.replace(tok, "").strip()
    return line


if __name__ == '__main__':
    import sys
    import glob
    from tqdm import tqdm
    from pprint import pprint
    import json
    files = glob.glob(sys.argv[1])
    res = dict()
    for file in tqdm(files):
        res[file] = calculate_perplexity_for_file(file)
        
    
    # sort by perplexity
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1]['ppl'])}
    
    for file in res:
        # show a few lines
        print(f"File: {file}")
        pprint(res[file]['data'][:5])
        
        # show the perplexity
        print(f"Perplexity: {res[file]['ppl']}")
        print(f"Percentage of unique lines: {res[file]['perc_unique_lines']}")
        print(f"Percentage of unique tokens: {res[file]['perc_unique_tokens']}")
        print("-" * 100)
    
    # Create a nice MARKDOWN report with: i) sample sentences, ii) perplexity, iii) percentage of unique lines, iv) percentage of unique tokens
    
    import random
    print("| File | Sample Sentences | Perplexity | % Unique Lines | % Unique Tokens |")
    for file in res:
        sentences = set(res[file]['data'])
        # pick 5 random sentences
        sentences = random.sample(sentences, 5) if len(sentences) > 5 else sentences
        
        filename = "#".join(file.split("/")[:-1])
        # print row
        print('-' * 80)
        if res[file]['perc_unique_tokens'] > 0:
            print(f"| {filename} | {', '.join(sentences)} | {res[file]['ppl']} | {res[file]['perc_unique_lines']} | {res[file]['perc_unique_tokens']} |")
        
    
    with open("perplexity.json", "w") as f:
        json.dump(res, f)
    
    