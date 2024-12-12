import numpy as np
from elfragmentadonnx.model import OnnxSequenceTensorConverter


def test_conversion():
    conv = OnnxSequenceTensorConverter()
    max_len = conv.max_length
    foo = conv.convert("MYPEPTIDEK", 2, 32.0)

    assert foo["input_ids_s"].shape == (max_len,)
    assert foo["position_ids_s"].shape == (max_len,)
    assert foo["charge_ce_2"].shape == (2,)
    assert foo["src_key_padding_mask_s"].shape == (max_len,)

    assert np.allclose(foo["charge_ce_2"], np.array([2.0, 32.0]))
    assert np.allclose(foo["src_key_padding_mask_s"][:5], np.array([0.0] * 5))
    assert np.allclose(foo["src_key_padding_mask_s"][-5:], np.array([-np.inf] * (5)))
