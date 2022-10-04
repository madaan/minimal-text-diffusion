import logging
import os
import pathlib
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from src.utils.custom_tokenizer import create_tokenizer

logging.basicConfig(level=logging.INFO)

# BAD: this should not be global
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = create_tokenizer()


def create_or_load_embeddings(embed_dim: int, checkpoint_path: str):
    """
    Check if the embeddings and vocab already exist, if not, creates them using the sentence_list.

    """

    if pathlib.Path(f"{checkpoint_path}/random_emb.torch").exists():
        embeddings = load_embeddings(embed_dim, checkpoint_path)
    else:
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        embeddings = torch.nn.Embedding(tokenizer.vocab_size, embed_dim)
        print("initializing the random embeddings", embeddings)
        torch.nn.init.normal_(embeddings.weight)
        path_save = f"{checkpoint_path}/random_emb.torch"
        print(f"save the random encoder to {checkpoint_path}/random_emb.torch")
        torch.save(embeddings.state_dict(), path_save)

        path_save = f"{checkpoint_path}/random_emb.torch"
        if not os.path.exists(path_save):
            torch.save(embeddings.state_dict(), path_save)

    return embeddings


def load_embeddings(emb_dim, checkpoint_path):

    model = torch.nn.Embedding(tokenizer.vocab_size, emb_dim)
    path_save = f"{checkpoint_path}/random_emb.torch"
    model.load_state_dict(torch.load(path_save))
    return model


def get_dataloader(embeddings, data_path, batch_size):
    dataset = TextDataset(tokenizer=tokenizer, data_path=data_path, embeddings=embeddings)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle=True,
        num_workers=1,
        collate_fn=TextDataset.collate_pad,
    )

    while True:
        for batch in dataloader:
            yield batch


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        embeddings: torch.nn.Embedding,
        data_path: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.read_data()

    def read_data(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_csv(self.data_path, sep="\t", header=None)  # read text file
        logging.info(f"Tokenizing {len(data)} sentences")

        self.text = data[0].apply(lambda x: x.strip()).tolist()
        # encoded_input = self.tokenizer(self.questions, self.paragraphs)
        
        # check if tokenizer has a method 'encode_batch'
        if hasattr(self.tokenizer, 'encode_batch'):

            encoded_input = self.tokenizer.encode_batch(self.text)
            self.input_ids = [x.ids for x in encoded_input]
        
        else:
            encoded_input = self.tokenizer(self.text)
            self.input_ids = encoded_input["input_ids"]

        # list of list of int. Each with a different length depending on the sentence

        # print(len(self.input_ids))
        # print(self.input_ids[0])
        # self.hidden_states = []
        # with torch.no_grad():
        #     for i in tqdm(range(len(self.input_ids))):
        #         arr = np.array(self.embeddings(torch.tensor(self.input_ids[i])).cpu().tolist())
        #         self.hidden_states.append(arr)

        # assert len(self.input_ids) == len(self.hidden_states)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        with torch.no_grad():
            arr = np.array(self.embeddings(torch.tensor(self.input_ids[i])).cpu().tolist())
        # arr = np.array(self.hidden_states[i], dtype=np.float32)
        out_dict = {
            "input_ids": self.input_ids[i],
            # "attention_mask": [1] * len(self.input_ids[i]),
        }
        return arr, out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int = 256):
        max_token_len = 0
        num_elems = len(batch)
        embed_dim = batch[0][0].shape[-1]
        # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i][1]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        embeddings = torch.zeros(num_elems, max_token_len, embed_dim).float()

        for i in range(num_elems):
            toks = batch[i][1]["input_ids"]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            embeddings[i, :length] = torch.Tensor(batch[i][0])

        return embeddings, {"input_ids": tokens, "attention_mask": tokens_mask}
