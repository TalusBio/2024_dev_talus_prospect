from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from loguru import logger
from onnxruntime import InferenceSession
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from elfragmentador_train.data_utils import (
    collate_tf_inputs,
    make_src_key_padding_mask_torch,
)


class AAEmbedder(nn.Module):
    """Embeds sequences of aminacids.

    The input sequence has shape (N, S) where N is the batch size and S is the
    sequence length. The output is a tensor of shape (N, S, E) where E is the
    embedding dimension.

    For the sake of simplicity, the inputs will be integers in the range of 0-255.
    MOST are not used but aminoacids are encoded by their unsigned 8 bit integer
    representation. In other words: A: 65, C: 67, D: 68, E: 69, F: 70, G: 71 ...

    The termini are represented with the symbol for "^" and "$" respectively.
    Which are 94 and 36 respectively.

    representation for -> [ord(x) for x in "^MYPEPTIDEK$"]
    will be -> [94, 77, 89, 80, 69, 80, 84, 73, 68, 69, 75, 36]

    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(255, embedding_dim)

    def forward(self, x_ns: torch.Tensor) -> torch.Tensor:  # noqa: ANN201, D102
        """Embeds a batch of sequences.

        Args:
            x (torch.Tensor): A tensor of shape (N, S) where N is the batch size
                and S is the sequence length.

        Returns
        -------
            torch.Tensor: A tensor of shape (N, S, E) where E is the embedding
                dimension.

        """
        return self.embedding(x_ns)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding with learnable phases and frequencies.

    Inputs are expected to be of shape (N, S) where N is the batch size and S is
    the sequence length.

    The values are expected to be the position in the aminoacid sequence as a float.
    In other words the positions for the tokens in the peptide "MYPEPTIDEK" are

    >>> [f"{x}: {float(i)}" for i, x in enumerate("^MYPEPTIDEK$")]
    [
        '^: 0.0', 'M: 1.0', 'Y: 2.0', 'P: 3.0', 'E: 4.0', 'P: 5.0',
        'T: 6.0', 'I: 7.0', 'D: 8.0', 'E: 9.0', 'K: 10.0', '$: 11.0'
    ]
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.phase_proj = nn.Linear(1, embedding_dim)

    def forward(self, x_ns: torch.Tensor) -> torch.Tensor:  # noqa: ANN201, D102
        out_nse = torch.sin(self.phase_proj(x_ns.unsqueeze(-1)))
        return out_nse


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""

    hidden_size: int
    num_layers: int
    num_attention_heads: int
    dropout: float
    mlp_hidden_size: int = 512
    mlp_output_size: int = 4


class TransformerModel(nn.Module):  # noqa: D101
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = AAEmbedder(config.hidden_size)
        self.position_embeddings = PositionalEncoding(config.hidden_size)
        self.charge_encoder = nn.Linear(1, config.hidden_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        transformer_encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self.transformer_encoder = transformer_encoder

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_size, config.mlp_output_size),
        )

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def clone_in_cpu(self) -> TransformerModel:
        """Clones the model in CPU.

        Makes a copy of the model, transfers the weights and puts it in CPU.
        """
        out = TransformerModel(self.config).cpu()
        out.eval()
        state = {k: v.cpu() for k, v in self.state_dict().items()}
        out.load_state_dict(state)
        return out.to("cpu")

    def to_onnx(self, path: str) -> None:
        """Exports the model to ONNX.

        Parameters
        ----------
            path (str): The path to export the model to.

        Raises
        ------
            RuntimeError: If the model fails to export.

        """
        model = self.clone_in_cpu()

        # test_inputs = _test_inputs(padding_size=20)
        # seq_len = test_inputs["position_ids_ns"].shape[1]

        test_inputs = _sample_inputs_batch()

        # Create a temporary directory to store the exported model
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Export the model to the temporary directory
            model_path = os.path.join(tmpdirname, "model.onnx")
            torch.onnx.export(
                model=model,
                f=model_path,
                args=tuple(test_inputs.values()),
                input_names=tuple(test_inputs.keys()),
                output_names=["output_ne"],
                export_params=True,
                dynamo=True,
                verify=True,
                verbose=True,
            )

            # Load the model from the temporary directory

            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)

            # Create an inference session using the ONNXRuntime
            session = InferenceSession(model_path)

            # Run inference using the ONNXRuntime
            test_inputs = _sample_inputs_batch()
            local_seq_len = test_inputs["position_ids_ns"].shape[1]
            batch_size = test_inputs["input_ids_ns"].shape[0]
            output = session.run(None, {k: v.numpy() for k, v in test_inputs.items()})
            if output[0].shape != (
                batch_size,
                local_seq_len,
                self.config.mlp_output_size,
            ):
                msg = f"Erroneous onnx export, expected (batch ({batch_size}),"
                msg += f" seq_len ({local_seq_len}),"
                msg += f" {self.config.mlp_output_size}) output shape"
                msg += f" but got {output[0].shape}"
                raise RuntimeError(msg)

        # once it passes export out of the temporary directory
        logger.info(f"Exporting to {path}")
        shapes = {k: v.shape for k, v in test_inputs.items()}
        logger.info(f"test_inputs: {shapes}")
        torch.onnx.export(
            model=model,
            f=path,
            args=tuple(test_inputs.values()),
            input_names=tuple(test_inputs.keys()),
            output_names=["output_ne"],
            export_params=True,
            dynamo=True,
            verify=True,
            verbose=True,
        )

    def forward(  # noqa: ANN201, D102
        self,
        input_ids_ns: torch.Tensor,
        position_ids_ns: torch.Tensor,
        charge_n1: torch.Tensor,
        src_key_padding_mask_ns: torch.Tensor,
    ):
        inputs_embeds_nse = self.embeddings(input_ids_ns)
        position_embeds_nse = self.position_embeddings(position_ids_ns)
        charge_embeds_ne = F.relu(self.charge_encoder(charge_n1))
        charge_embeds_nse = torch.einsum(
            "...se, ...e -> ...se",
            torch.ones_like(inputs_embeds_nse),
            charge_embeds_ne,
        )

        hidden_states_nse = inputs_embeds_nse + position_embeds_nse + charge_embeds_nse
        hidden_states_nse = self.layernorm(hidden_states_nse)
        hidden_states_nse = self.dropout(hidden_states_nse)

        hidden_states_nse = self.transformer_encoder(
            hidden_states_nse,
            src_key_padding_mask=src_key_padding_mask_ns,
        )
        output_nse = F.elu(self.mlp(hidden_states_nse)) + 1

        return output_nse


def _test_inputs(extra_seq_len: int = 0) -> dict[str, torch.Tensor]:
    """A single entry used as input for testing of the model."""
    input_str = "^MYPEPT" + ("A" * extra_seq_len) + "IDEK$" + (" " * extra_seq_len)

    input_ids_ns = torch.tensor(
        [ord(x) for x in input_str],
        dtype=torch.long,
        device="cpu",
    ).unsqueeze(0)
    position_ids_ns = torch.arange(len(input_str), device="cpu").float().unsqueeze(0)

    return {
        "input_ids_ns": input_ids_ns,
        "position_ids_ns": position_ids_ns,
        "charge_n1": torch.tensor([[2.0]], dtype=torch.float, device="cpu"),
        "src_key_padding_mask_ns": make_src_key_padding_mask_torch(
            input_ids_ns,
            pad_token_id=ord(" "),
        ),
    }


def _test_inputs_unbatched(extra_seq_len: int = 0) -> dict[str, torch.Tensor]:
    """A sample input used as input for testing of the model.

    Check the `_sample_inputs_batch` function for a sample batch.
    """
    return {k: v[0] for k, v in _test_inputs(extra_seq_len).items()}


def _sample_inputs_batch() -> dict[str, torch.Tensor]:
    """A sample batch used as input for testing of the model."""
    batch = [
        _test_inputs_unbatched(extra_seq_len=pad_size) for pad_size in range(1, 10)
    ]
    return collate_tf_inputs(batch)
