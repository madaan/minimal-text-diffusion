import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

from src.utils.custom_tokenizer import create_tokenizer

logging.basicConfig(level=logging.INFO)

# BAD: this should not be global
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = create_tokenizer()




def get_dataloader(data_path, batch_size):
    dataset = TextDataset(tokenizer=tokenizer, data_path=data_path)

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
        data_path: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
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

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        out_dict = {
            "input_ids": self.input_ids[i],
            # "attention_mask": [1] * len(self.input_ids[i]),
        }
        return out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int = 256):
        max_token_len = 0
        num_elems = len(batch)
        # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()

        for i in range(num_elems):
            toks = batch[i]["input_ids"]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1

        return None, {"input_ids": tokens, "attention_mask": tokens_mask}
