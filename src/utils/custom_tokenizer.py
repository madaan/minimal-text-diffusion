import json
import logging
import pathlib
import torch
from transformers import AutoTokenizer

from tokenizers.processors import BertProcessing
from tokenizers import ByteLevelBPETokenizer, decoders

logging.basicConfig(level=logging.INFO)

def create_tokenizer(return_pretokenized, path, tokenizer_type: str = "word-level"):
    if return_pretokenized:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer

    if tokenizer_type == "byte-level":
        return read_byte_level(path)
    elif tokenizer_type == "word-level":
        return read_word_level(path)
    else:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")

def train_bytelevel(
    path,
    vocab_size=10000,
    min_frequency=1,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
):

    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=[path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.save_model(str(pathlib.Path(path).parent))



def read_byte_level(path: str):
    tokenizer = ByteLevelBPETokenizer(
        f"{path}/vocab.json",
        f"{path}/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=512)

    print(
        tokenizer.encode(
            "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
        ).tokens
    )

    with open(f"{path}/vocab.json", "r") as fin:
        vocab = json.load(fin)

    # add length method to tokenizer object
    tokenizer.vocab_size = len(vocab)

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.decoder = decoders.ByteLevel()
    print(tokenizer.vocab_size)

    print(
        tokenizer.encode(
            "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
        ).ids
    )

    print(
        tokenizer.decode(
            tokenizer.encode(
                "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
            ).ids,
            skip_special_tokens=True,
        )
    )

    ids = tokenizer.encode(
        "Bores can be divided into two classes; those who have their own particular subject, and those who do not need a subject."
    ).ids
    tensor = torch.tensor(ids)
    print(tokenizer.decode(tensor.tolist(), skip_special_tokens=True))
    print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def read_word_level(path: str):

    from transformers import PreTrainedTokenizerFast

    logging.info(f"Loading tokenizer from {path}/word-level-vocab.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{str(pathlib.Path(path))}/word-level-vocab.json",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        padding_side="right",
    )

    # add length property to tokenizer object
    tokenizer.__len__ = property(lambda self: self.vocab_size)

    return tokenizer


def train_word_level_tokenizer(
    path: str,
    vocab_size: int = 10000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
):

    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    from tokenizers.pre_tokenizers import Digits, Whitespace
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import WordLevelTrainer

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Digits(individual_digits=True), Whitespace()]
    )
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )

    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=[path], trainer=trainer)

    tokenizer.__len__ = property(lambda self: self.vocab_size)

    tokenizer.enable_truncation(max_length=512)

    print(tokenizer.encode("the red.").ids)

    print(tokenizer.encode("the red."))

    tokenizer.save(f"{str(pathlib.Path(path).parent)}/word-level-vocab.json")


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "train-word-level":
        train_word_level_tokenizer(path=sys.argv[2])
    elif sys.argv[1] == "train-byte-level":
        train_bytelevel(path=sys.argv[2])
    elif sys.argv[1] == "create":
        create_tokenizer(path=sys.argv[2])
