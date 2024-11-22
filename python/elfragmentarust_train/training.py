import torch
import lightning as L
import argparse
from elfragmentarust_train.transformer_model import TransformerModel, TransformerConfig
from elfragmentarust_train.fragment_dataset import (
    FragmentationDatasetFactory,
    SequenceTensorConverter,
    ef_batch_collate_fn,
    _testing_row,
)
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
from pathlib import Path
import os
from .data_utils import batch_to_inputs
from dataclasses import asdict
from collections import defaultdict
import json


class FragmentationModel(L.LightningModule):
    def __init__(
        self,
        model: TransformerModel,
        ds_factory: FragmentationDatasetFactory,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.ds_factory = ds_factory
        self.batch_size = batch_size
        self.token_count = defaultdict(int)

        self.save_hyperparameters(asdict(self.model.config))

    def batch_forward(
        self, batch: dict[str, torch.Tensor], add_counts=True
    ) -> torch.Tensor:
        inputs = batch_to_inputs(batch)
        # Add the token count to the inputs

        if add_counts:
            counts = inputs["input_ids_ns"].unique(return_counts=True)
            for k, v in zip(counts[0], counts[1]):
                self.token_count[k.item()] += v.item()

        return self.model(**inputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = self.model(*args, **kwargs)
        return outputs

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.batch_forward(batch, add_counts=True)
        bo = self.batch_to_outputs(batch)

        loss = self.loss_fn(outputs, bo, mask=True, show=batch_idx % 200 == 0)
        self.log("train_loss", loss, prog_bar=True)

        ## Other random stuff ...
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.batch_forward(batch, add_counts=False)
        bo = self.batch_to_outputs(batch)

        loss = self.loss_fn(outputs, bo, mask=False, show=batch_idx % 200 == 0)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # AdamW + onecycle
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10.0,
            total_steps=self.trainer.estimated_stepping_batches,
            final_div_factor=500.0,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        return self.ds_factory.get_train_ds().with_dataloader(
            batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.ds_factory.get_val_ds().with_dataloader(
            batch_size=self.batch_size, shuffle=False
        )

    def on_train_epoch_end(self):
        print(f"Token counts: {self.token_count}")
        epoch_no = self.trainer.current_epoch
        out_loc = os.path.join(self.trainer.log_dir, f"token_counts_{epoch_no}.json")
        with open(out_loc, "w") as f:
            json.dump(self.token_count, f, indent=2)

    def on_validation_epoch_end(self):
        epoch_no = self.trainer.current_epoch
        out_loc = os.path.join(self.trainer.log_dir, f"model_{epoch_no}.onnx")
        self.model.to_onnx(str(out_loc))

    @staticmethod
    def batch_to_outputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch["intensity_tensor"]

    @staticmethod
    def loss_fn(predictions, outputs, mask=True, show=False):
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
            lp = (predictions[0] * 100)[:15, :].long()
            lg = (outputs[0] * 100)[:15, :].long()
            print(f"preds: \n{lp}")
            print(f"gt_outs: \n{lg}")

        loss = F.mse_loss(predictions, outputs)
        return loss


def build_parser():
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


def build_trainer():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        filename="{epoch}-{val_loss:.6f}",
    )
    return L.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        max_time={"days": 0, "hours": 12},
        enable_progress_bar=True,
        limit_val_batches=0.5,
        gradient_clip_val=1.0,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    fragments_dir = Path(args.data_dir) / "fragments_pq"
    precursor_dir = Path(args.data_dir) / "precursors_pq"

    if (not fragments_dir.exists()) or (not precursor_dir.exists()):
        raise ValueError(
            f"Data directories do not exist: {fragments_dir} or {precursor_dir}"
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
    print(f"Building model with config: \n{model_config}")
    model = TransformerModel(model_config)
    lit_model = FragmentationModel(model, factory, args.batch_size, args.lr)
    trainer = build_trainer()
    if args.weights_from_checkpoint:
        # Load only the model weights
        print(f"Loading weights from checkpoint: {args.weights_from_checkpoint}")
        sd = {
            k: v.cpu()
            for k, v in torch.load(args.weights_from_checkpoint)["state_dict"].items()
        }
        lit_model.load_state_dict(sd)
    trainer.fit(lit_model)


def _sample_batch():
    conv = SequenceTensorConverter()
    sample_row = _testing_row()
    sample_row2 = _testing_row()
    sample_row2["precursor_charge"] = 3
    sample_row2["peptide_sequence"] += "K"
    tensors = conv.convert(sample_row)
    tensors2 = conv.convert(sample_row2)
    ef_batch = ef_batch_collate_fn([tensors, tensors2])
    return ef_batch


def _test_model():
    config = TransformerConfig(
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        dropout=0.1,
    )
    model = TransformerModel(config)
    return model


def test_smoke_batch_to_inputs():
    batch = _sample_batch()
    model = _test_model()

    lit_model = FragmentationModel(model, None)
    inputs = batch_to_inputs(batch)
    outputs = model(**inputs)
    assert outputs.shape[0] == 2
    assert outputs.shape[2] == 4
    assert outputs.shape[1] == inputs["input_ids_ns"].shape[1]

    _loss = lit_model.training_step(batch, 0)


if __name__ == "__main__":
    main()
