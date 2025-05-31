from typing import Tuple
from torch import nn, Tensor
import torch
from torchvision.models.densenet import DenseNet  # type: ignore
from positional_encodings.torch_encodings import (  # type: ignore
    PositionalEncodingPermute2D,
    PositionalEncodingPermute1D,
    Summer,
)


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        activation: str = "relu",
        num_dec_layers: int = 3,
        growth_rate: int = 24,
        dense_block_config: Tuple[int, ...] = (16, 16, 16),
    ) -> None:
        super(Model, self).__init__()

        self.encoder = DenseNet(
            growth_rate=growth_rate,
            block_config=dense_block_config,  # type: ignore
        )

        self.feature_proj = nn.Conv2d(
            self.encoder.classifier.in_features, d_model, kernel_size=1
        )

        self.pos_enc_2d_summer = Summer(PositionalEncodingPermute2D(d_model))

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_enc_1d_summer = Summer(PositionalEncodingPermute1D(d_model))

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_dec_layers,
        )

        self.classifier = nn.Linear(d_model, vocab_size)

    def encode(self, img: Tensor) -> Tensor:
        # (batch_size, num_channels, height, width)

        x = img.type(torch.float32)
        x = self.encoder.features(x)
        # (batch_size, num_features, height, width

        x = self.feature_proj(x)
        x = self.pos_enc_2d_summer(x)
        # (batch_size, d_model, height, width)

        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # (batch_size, seq_len, d_model)

        return x

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None,
        tgt_key_padding_mask: Tensor | None,
    ) -> Tensor:
        # (batch_size, seq_len)
        # (batch_size, seq_len, d_model)
        # (seq_len, seq_len)

        tgt = self.embedding(tgt)
        tgt = self.pos_enc_1d_summer(tgt)
        # (batch_size, seq_len, d_model)

        out: Tensor = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        # (batch_size, seq_len, d_model)

        out = self.classifier(out)
        # (batch_size, seq_len, vocab_size)

        return out

    def forward(
        self,
        img: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # (batch_size, num_channels, height, width)
        # (batch_size, seq_len)
        # (seq_len, seq_len)

        memory = self.encode(img)
        # (batch_size, seq_len, d_model)

        out = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask)
        # (batch_size, seq_len, vocab_size)

        return out

    def beam_search(
    self,
    x: Tensor,
    beam_size: int = 5,
    max_len: int = 50,
    sos_token: int = 1,
    eos_token: int = 2,
) -> Tensor:
    self.eval()

    with torch.no_grad():
        memory = self.encode(x)  # (1, seq_len, d_model)

        sequences = [[sos_token]]
        scores = torch.zeros(1, device=x.device)

        for _ in range(max_len):
            all_candidates = []

            for i, seq in enumerate(sequences):
                if seq[-1] == eos_token:
                    all_candidates.append((scores[i], seq))
                    continue

                tgt = torch.tensor([seq], device=x.device)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(x.device)

                logits = self.decode(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
                logits = logits[:, -1, :]  # (1, vocab_size)
                log_probs = torch.log_softmax(logits, dim=-1)

                topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

                for log_prob, token_id in zip(topk_log_probs[0], topk_indices[0]):
                    new_seq = seq + [token_id.item()]
                    new_score = scores[i] + log_prob
                    all_candidates.append((new_score, new_seq))

            # Keep top beam_size sequences
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            sequences = [seq for _, seq in all_candidates[:beam_size]]
            scores = torch.tensor([score for score, _ in all_candidates[:beam_size]], device=x.device)

            # Early stop if all sequences have ended
            if all(seq[-1] == eos_token for seq in sequences):
                break

        # Return best sequence (highest score)
        return torch.tensor(sequences[0], device=x.device)
