from functools import partial
from typing import List, Tuple
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # type: ignore
from transformers import PreTrainedTokenizerFast  # type: ignore
import torch.nn.functional as F


def pad_img_to_size(
    img: Tensor, target_height: int, target_width: int, pad_value: int = 255
) -> Tensor:
    _, h, w = img.shape

    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left

    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)


def load_tokenizer(file: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file)

    tokenizer.pad_token = "<pad>"  # type: ignore
    tokenizer.add_special_tokens({"bos_token": "<sos>", "eos_token": "<eos>"})  # type: ignore

    return tokenizer


def collate_fn(
    batch: List[Tuple[Tensor, str]],
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Tensor, Tensor, Tensor]:
    imgs, formulas = zip(*batch)

    max_height = max(img.shape[1] for img in imgs)
    max_width = max(img.shape[2] for img in imgs)

    # Pad images to the same size
    padded_imgs = [pad_img_to_size(img, max_height, max_width) for img in imgs]
    img_batch: Tensor = torch.stack(padded_imgs)  # (B, C, H, W)

    # Tokenize with padding
    tokenized = tokenizer(
        list(formulas),
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,  # optional: avoid very long formulas
    )  # type: ignore

    input_ids: Tensor = tokenized["input_ids"]  # (B, T) # type: ignore
    attn_mask: Tensor = tokenized["attention_mask"]  # (B, T) # type: ignore

    return img_batch, input_ids, attn_mask


class LatexEquationDataset(Dataset[Tuple[Tensor, str]]):
    def __init__(self) -> None:
        self.data = load_dataset(  # type: ignore
            "OleehyO/latex-formulas", "cleaned_formulas"
        ).with_format("torch")["train"]

        self.tokenizer = load_tokenizer("latex-tokenizer.json")
        self.vocab_size = self.tokenizer.vocab_size  # type: ignore

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.data[idx]["image"], self.data[idx]["latex_formula"]

    def get_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer),
        )
