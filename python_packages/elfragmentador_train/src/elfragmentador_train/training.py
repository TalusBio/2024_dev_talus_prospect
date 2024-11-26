import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from loguru import logger
from torch.utils.data import DataLoader

from elfragmentador_train.fragment_dataset import (
    FragmentationDatasetFactory,
)
from elfragmentador_train.transformer_model import TransformerConfig, TransformerModel

from .data_utils import batch_to_inputs


class FragmentationModel(L.LightningModule):  # noqa: D101
    def __init__(
        self,
        model: TransformerModel,
        ds_factory: FragmentationDatasetFactory,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.ds_factory = ds_factory
        self.batch_size = batch_size
        self.token_count = defaultdict(int)

        self.save_hyperparameters(asdict(self.model.config))

    def batch_forward(
        self,
        batch: dict[str, torch.Tensor],
        add_counts: bool = True,
    ) -> torch.Tensor:
        """Forward pass for the batch.

        Parameters
        ----------
            batch (dict[str, torch.Tensor]): The batch.
            add_counts (bool, optional): Whether to add the token counts to the inputs.
                Defaults to True.

        Returns
        -------
            torch.Tensor: The outputs.
        """
        inputs = batch_to_inputs(batch)
        # Add the token count to the inputs

        if add_counts:
            counts = inputs["input_ids_ns"].unique(return_counts=True)
            for k, v in zip(counts[0], counts[1], strict=False):
                self.token_count[k.item()] += v.item()

        return self.model(**inputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # noqa: ANN201, D102, ANN003, ANN002
        outputs = self.model(*args, **kwargs)
        return outputs

    def training_step(  # noqa: D102
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        outputs = self.batch_forward(batch, add_counts=True)
        bo = self.batch_to_outputs(batch)

        loss = self.loss_fn(outputs, bo, mask=True, show=batch_idx % 200 == 0)
        self.log("train_loss", loss, prog_bar=True)

        ## Other random stuff ...
        try:
            cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", cur_lr, prog_bar=True, on_step=True)
        except IndexError:
            logger.warning("No optimizers found")

        return loss

    def validation_step(  # noqa: D102
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        outputs = self.batch_forward(batch, add_counts=False)
        bo = self.batch_to_outputs(batch)

        loss = self.loss_fn(outputs, bo, mask=False, show=batch_idx % 200 == 0)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(  # noqa: D102
        self,
    ) -> tuple[
        torch.optim.Optimizer,
        list[dict[str, torch.optim.lr_scheduler.OneCycleLR | str]],
    ]:
        # AdamW + onecycle
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10.0,
            total_steps=self.trainer.estimated_stepping_batches,
            final_div_factor=500.0,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:  # noqa: ANN201, D102
        return self.ds_factory.get_train_ds().with_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: ANN201, D102
        return self.ds_factory.get_val_ds().with_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
        )

    def on_train_epoch_end(self) -> None:  # noqa: ANN201, D102
        epoch_no = self.trainer.current_epoch
        out_loc = os.path.join(self.trainer.log_dir, f"token_counts_{epoch_no}.json")
        with open(out_loc, "w") as f:
            json.dump(self.token_count, f, indent=2)

    def on_validation_epoch_end(self) -> None:  # noqa: ANN201, D102
        epoch_no = self.trainer.current_epoch
        out_loc = os.path.join(self.trainer.log_dir, f"model_{epoch_no}.onnx")
        self.model.to_onnx(str(out_loc))

    @staticmethod
    def batch_to_outputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Extract the outputs from the batch."""
        return batch["intensity_tensor"]

    @staticmethod
    def loss_fn(
        predictions: torch.Tensor,
        outputs: torch.Tensor,
        mask: bool = True,
        show: bool = False,
    ) -> torch.Tensor:
        """Computes the loss function.

        Parameters
        ----------
            predictions (torch.Tensor): The predictions.
            outputs (torch.Tensor): The outputs.
            mask (bool, optional): Whether to mask the predictions. Defaults to True.
            show (bool, optional): Whether to show the predictions. Defaults to False.

        Returns
        -------
            torch.Tensor: The loss.

        """
        ## The predictions can be larger than the intensity tensor.
        ## So we trim dimension 1 to match
        predictions = predictions[:, : outputs.shape[1], :]

        if mask:
            randmask = (torch.rand_like(outputs) > 0.5).float()
            predictions = predictions * randmask
            outputs = outputs * randmask

        predictions = torch.nn.functional.normalize(predictions, dim=(1, 2))
        outputs = torch.nn.functional.normalize(outputs, dim=(1, 2))

        if show:
            logger.info((predictions[0] * 100)[:15, :].long())
            logger.info((outputs[0] * 100)[:15, :].long())

        loss = F.mse_loss(predictions, outputs)
        return loss


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a Fragmentation Model",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data")
    parser.add_argument(
        "--partitions_keep",
        type=str,
        nargs="+",
        default=None,
        help="Partitions to keep",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weights_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load weights from",
    )
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Num layers")
    parser.add_argument("--num_attention_heads", type=int, default=2, help="Num heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument(
        "--mlp_hidden_size",
        type=int,
        default=124,
        help="MLP hidden size",
    )
    return parser


def build_trainer() -> L.Trainer:
    """Builds the trainer."""
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        filename="{epoch}-{val_loss:.6f}",
    )
    modelsum = RichModelSummary(max_depth=2)
    return L.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, modelsum],
        max_time={"days": 0, "hours": 24},
        enable_progress_bar=True,
        limit_val_batches=0.5,
        gradient_clip_val=1.0,
    )


def main() -> None:
    """Main function for training the model."""
    parser = build_parser()
    args = parser.parse_args()

    fragments_dir = Path(args.data_dir) / "fragments_pq"
    precursor_dir = Path(args.data_dir) / "precursors_pq"

    if (not fragments_dir.exists()) or (not precursor_dir.exists()):
        raise ValueError(
            f"Data directories do not exist: {fragments_dir} or {precursor_dir}",
        )
    factory = FragmentationDatasetFactory(
        fragments_dir,
        precursor_dir,
        partitions_keep=args.partitions_keep,
    )

    model_config = TransformerConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_output_size=4,
    )
    model = TransformerModel(model_config)
    lit_model = FragmentationModel(model, factory, args.batch_size, args.lr)
    trainer = build_trainer()
    if args.weights_from_checkpoint:
        # Load only the model weights
        sd = {
            k: v.cpu()
            for k, v in torch.load(args.weights_from_checkpoint)["state_dict"].items()
        }
        lit_model.load_state_dict(sd)
    trainer.fit(lit_model)


if __name__ == "__main__":
    main()
