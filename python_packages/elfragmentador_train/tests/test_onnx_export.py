from elfragmentador_train.transformer_model import (
    TransformerModel,
    TransformerConfig,
    _test_inputs,
)


def _testing_model():
    config = TransformerConfig(
        hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        mlp_output_size=2,
    )

    model = TransformerModel(config)
    model.eval()
    return model, config


def test_transformer_model():
    model, config = _testing_model()
    input_dict = _test_inputs()
    output_ne = model(**input_dict)
    seq_len = input_dict["position_ids_ns"].shape[1]
    assert output_ne.shape == (1, seq_len, 2)


def test_onnx_export_import():
    model, config = _testing_model()
    model.to_onnx("pytest_model.onnx")
