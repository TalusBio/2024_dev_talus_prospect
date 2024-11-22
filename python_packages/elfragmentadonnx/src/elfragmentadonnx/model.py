from onnxruntime import InferenceSession
from elfragmentador_core.config import IntensityTensorConfig
from elfragmentador_core.converter import SequenceTensorConverter
from elfragmentador_core.data_utils import make_src_key_padding_mask
import numpy as np


class OnnxSequenceTensorConverter(SequenceTensorConverter):
    max_length: int = 30

    def convert(self, modified_sequence, charge):
        tokens, positions = self.tokenize_proforma(
            modified_sequence,
            padded_length=self.max_length,
        )
        charge_tensor = np.array([charge], dtype=np.float32)
        padding_mask = make_src_key_padding_mask(tokens, pad_token_id=ord(" "))

        assert charge_tensor.shape == (1,)
        assert padding_mask.shape == (self.max_length,)
        assert tokens.shape == (self.max_length,)
        assert positions.shape == (self.max_length,)

        return {
            "input_ids_s": tokens,
            "position_ids_s": positions,
            "charge_1": charge_tensor,
            "src_key_padding_mask_s": padding_mask,
        }


class OnnxPeptideTransformer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.converter = OnnxSequenceTensorConverter()
        self.session = None

        # session = InferenceSession(model_path)
        # output = session.run(None, {k: v.numpy() for k, v in test_inputs.items()})
