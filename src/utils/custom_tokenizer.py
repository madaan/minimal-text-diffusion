import json
import torch
from transformers import AutoTokenizer

from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, decoders


def train():
    paths = ["data/author-quote.txt"]

    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=5000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model("data/author-quote/")


def create_tokenizer(return_pretokenized=True):
    if return_pretokenized:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer

    tokenizer = ByteLevelBPETokenizer(
        "data/author-quote/vocab.json",
        "data/author-quote/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=512)

    print(tokenizer.encode("Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject.").tokens)

    with open("data/author-quote/vocab.json", "r") as fin:
        vocab = json.load(fin)
    
    # add length method to tokenizer object
    tokenizer.vocab_size = len(vocab)
    
    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.decoder = decoders.ByteLevel()
    print(tokenizer.vocab_size)

    print(tokenizer.encode("Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject.").ids)
    
    print(tokenizer.decode(tokenizer.encode("Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject.").ids, skip_special_tokens=True))

    ids = tokenizer.encode("Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject.").ids
    tensor = torch.tensor(ids)
    print(tokenizer.decode(tensor.tolist(), skip_special_tokens=True))
    
    return tokenizer  


if __name__ == "__main__":
    create_tokenizer()