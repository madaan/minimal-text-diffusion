import logging
import math
import os
import pathlib
from typing import List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# sys.path.insert(0, os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise
from collections import Counter, defaultdict
import torch, json
from spacy.lang.en import English


# def dataloader_from_text(
#     data_txt_path: str,
#     sequence_length: int,
#     checkpoint_path: str,
#     embed_dim: int,
#     batch_size: int,
#     tok_thresh: int,
# ):
#     tokenized_and_embedded_text, embeddings, vocab_dict = parse_data_to_embeddings(
#         data_txt_path, sequence_length, checkpoint_path, embed_dim, tok_thresh
#     )
#     dataloader = get_dataloader(
#         tokenized_and_embedded_text, sequence_length, batch_size
#     )
#     return dataloader, embeddings, vocab_dict


def parse_data_to_embeddings(
    txt_file_path: str,
    seqlen: int,
    checkpoint_path: str,
    embed_dim: int,
    tok_thresh: int,
):


    sentence_list = tokenize_txt_file(txt_file_path)
    embeddings, vocab_dict, tokenizer = create_or_load_embeddings_and_vocab(embed_dim=embed_dim, checkpoint_path=checkpoint_path, sentence_list=sentence_list,
    tok_frequency_thresh=tok_thresh)


    result_train_lst = helper_tokenize_encode(
        sentence_list=sentence_list, vocab_dict=vocab_dict, seqlen=seqlen, embeddings=embeddings
    )

    return {"train": result_train_lst}, embeddings, vocab_dict



def tokenize_txt_file(data_txt_path: str) -> List[List[str]]:
    nlp = English()
    tokenizer = nlp.tokenizer

    print(f"loading dataset from {data_txt_path}")
    sentence_list = []

    with open(data_txt_path, "r") as file_in:
        for line in file_in:
            word_lst = [token.text for token in tokenizer(line.strip().lower())]
            sentence_list.append(word_lst)
    return sentence_list


def create_or_load_embeddings_and_vocab(
    embed_dim: int, checkpoint_path: str, sentence_list: List[List[str]], tok_frequency_thresh: int
):
    """
    Check if the embeddings and vocab already exist, if not, creates them using the sentence_list.
    
    """

    if (
        pathlib.Path(f"{checkpoint_path}/vocab.json").exists()
        and pathlib.Path(f"{checkpoint_path}/random_emb.torch").exists()
    ):
        embeddings, vocab_dict, tokenizer = load_embeddings_and_vocab(
            embed_dim, checkpoint_path
        )
    else:
        vocab_dict = get_vocab(sentence_list, tok_frequency_thresh)
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        path_save_vocab = f"{checkpoint_path}/vocab.json"
        print(f"save the vocab to {path_save_vocab}")
        with open(path_save_vocab, "w") as f:
            json.dump(vocab_dict, f)

        embeddings = torch.nn.Embedding(len(vocab_dict), embed_dim)
        print("initializing the random embeddings", embeddings)
        torch.nn.init.normal_(embeddings.weight)
        path_save = f"{checkpoint_path}/random_emb.torch"
        print(f"save the random encoder to {checkpoint_path}/random_emb.torch")
        torch.save(embeddings.state_dict(), path_save)

        path_save = f"{checkpoint_path}/random_emb.torch"
        if not os.path.exists(path_save):
            torch.save(embeddings.state_dict(), path_save)
        tokenizer = {v: k for k, v in vocab_dict.items()}

    return embeddings, vocab_dict, tokenizer


def get_vocab(sentence_lst: List[str], tok_thresh: int) -> dict:

    counter = Counter()
    for input_ids in sentence_lst:
        counter.update(input_ids)

    vocab_dict = {"START": 0, "END": 1, "UNK": 2, "PAD": 3}
    for k, v in counter.items():
        if v > tok_thresh:
            vocab_dict[k] = len(vocab_dict)
    logging.info(f"vocab size: {len(vocab_dict)}")
    return vocab_dict


def load_embeddings_and_vocab(emb_dim, checkpoint_path):

    path_save_tokenizer = f"{checkpoint_path}/vocab.json"
    print(f"loading from {path_save_tokenizer}")
    with open(path_save_tokenizer, "r") as f:
        vocab = json.load(f)
    tokenizer = {v: k for k, v in vocab.items()}
    model = torch.nn.Embedding(len(tokenizer), emb_dim)
    path_save = f"{checkpoint_path}/random_emb.torch"
    model.load_state_dict(torch.load(path_save))

    return model, vocab, tokenizer


def get_dataloader(tokenized_and_embedded_text, sequence_length, batch_size):
    dataset = TextDataset(
        tokenized_and_embedded_text,
        resolution=math.sqrt(
            sequence_length
        ),  # this is just to maintain some of the jargon from the image-based codebases, see my comment on TextDataset
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )
    
    while True:
        for batch in dataloader:
            yield batch
    


def helper_tokenize_encode(sentence_list: List[List[int]], vocab_dict: dict, embeddings, seqlen: int):
    """
    Converts a list of sentences to a list of tokenized and encoded sentences.
    """
    
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for input_ids in sentence_list:
            tokenized_ = [vocab_dict.get(x, vocab_dict["UNK"]) for x in input_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst["word_ids"].append(input_ids)
        
        # if padding_mode == 'block':
        group_lst["word_ids"] = _collate_batch_helper(
            group_lst["word_ids"], vocab_dict["PAD"], seqlen
        )

        for input_ids in group_lst["word_ids"]:
            hidden_state = embeddings(torch.tensor(input_ids))
            result_train_lst.append(
                {"input_ids": input_ids, "hidden_states": hidden_state.cpu().tolist()}
            )

    return result_train_lst


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full(
        [len(examples), max_length], pad_token_id, dtype=torch.int64
    ).tolist()
    mask_ = torch.full(
        [len(examples), max_length], pad_token_id, dtype=torch.int64
    ).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


class TextDataset(Dataset):
    """
    A bunch of the code is ultimately salvaged from Glide, so there is some terminology that talks about images.
    A prime example is resolution: which makes no sense for text, but we set it to sqrt of the sequence length.
    """

    def __init__(
        self,
        text_datasets,
        resolution,
        model_arch="transformer",
        classes=None,
        shard=0,
        num_shards=1,
        eigen_transform=None,
        noise_level: float = 0.0,
        is_conditional_gen: bool = False,
        mapping_func=None,
        model_emb=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets["train"])
        self.model_arch = model_arch
        self.noise_level = noise_level
        self.is_conditional_gen = is_conditional_gen
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        arr = np.array(
            self.text_datasets["train"][idx]["hidden_states"], dtype=np.float32
        )
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform["mean"]
            arr = arr @ self.eigen_transform["map"]
            arr = arr.reshape(old_shape)

        if hasattr(self, "noise_level") and self.noise_level > 0:
            arr = arr + self.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

        out_dict = {}
        out_dict["input_ids"] = np.array(self.text_datasets["train"][idx]["input_ids"])
        if self.is_conditional_gen:
            out_dict["src_ids"] = np.array(self.text_datasets["train"][idx]["src_ids"])
            out_dict["src_mask"] = np.array(
                self.text_datasets["train"][idx]["src_mask"]
            )

        return arr, out_dict
