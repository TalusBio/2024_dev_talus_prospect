import lightning as L
from elfragmentador_train.fragment_dataset import (
    TorchSequenceTensorConverter,
    _testing_row,
    ef_batch_collate_fn,
)
from elfragmentador_train.training import (
    FragmentationModel,
    TransformerConfig,
    TransformerModel,
    batch_to_inputs,
)
from loguru import logger


def _sample_batch():
    conv = TorchSequenceTensorConverter()
    sample_row = _testing_row()
    sample_row2 = _testing_row()
    sample_row2["precursor_charge"] = 3
    sample_row2["modified_sequence"] += "K"
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


def test_smoke_batch_to_inputs() -> None:
    batch = _sample_batch()
    model = _test_model()

    lit_model = FragmentationModel(model, None)
    inputs = batch_to_inputs(batch)
    outputs = model(**inputs)
    assert outputs.shape[0] == 2
    assert outputs.shape[2] == 4
    assert outputs.shape[1] == inputs["input_ids_ns"].shape[1]

    def log_patch(self, *args, **kwargs):  # noqa: ANN002, ANN003
        logger.info(f"Logging: {args}, {kwargs}")

    lt = L.Trainer(max_epochs=1)
    lt.log = log_patch
    lit_model.log = log_patch
    lit_model.trainer = lt
    _loss = lit_model.training_step(batch, 0)
