from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from math2latex.model import Model


def generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    return mask.to(device)


def train(
    device: torch.device,
    model: Model,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_epochs: int,
    lr: float = 1e-4,
) -> None:
    model.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=0)  # assuming 0 = <pad>
    optimizer: Adam = Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        epoch_loss: float = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for imgs, tgt, attn_mask in progress:
            imgs = imgs.to(device)
            tgt = tgt.to(device)
            attn_mask = attn_mask.to(device)
            # (batch_size, num_channels, height, width)
            # (batch_size, seq_len + 1)
            # (batch_size, seq_len + 1)

            # Shift tgt to create input/output
            tgt_input: Tensor = tgt[:, :-1]
            tgt_output: Tensor = tgt[:, 1:]
            # (batch_size, seq_len)
            # (batch_size, seq_len)

            tgt_mask: Tensor = generate_square_subsequent_mask(
                tgt_input.shape[1], device
            )  # (seq_len, seq_len)

            # Create key padding mask from attention mask:
            # attention_mask: 1 for real tokens, 0 for padding -> invert it
            tgt_key_padding_mask = attn_mask[:, :-1] == 0  # (batch_size, seq_len)

            # Forward pass
            logits: Tensor = model(imgs, tgt_input, tgt_mask, tgt_key_padding_mask)
            # (batch_size, seq_len, vocab_size)

            # Reshape and compute loss
            loss: Tensor = criterion(
                logits.view(-1, logits.shape[-1]),  # (batch_size * seq_len, vocab_size)
                tgt_output.reshape(-1),  # (batch_size * seq_len)
            )
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(dataloader):.4f}")
