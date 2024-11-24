from onnxruntime import InferenceSession
from elfragmentador_core.config import IntensityTensorConfig
from elfragmentador_core.converter import SequenceTensorConverter
from elfragmentador_core.data_utils import make_src_key_padding_mask
import numpy as np
from pathlib import Path
from typing import Iterator, Generator
import rustyms
import itertools


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


class OnnxSequenceTensorConverter(SequenceTensorConverter):
    max_length: int = 30

    def convert(self, modified_sequence, charge):
        return self.convert_with_peptide(modified_sequence, charge)[0]

    def _convert_linear_peptide(
        self, peptide: rustyms.LinearPeptide, charge: int
    ) -> dict[str, np.array]:
        tokens, positions = self.tokenize_linear_peptide(
            peptide, padded_length=self.max_length
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

    def convert_with_peptide(
        self, modified_sequence: str, charge: int
    ) -> tuple[dict[str, np.array], rustyms.LinearPeptide]:
        peptide = rustyms.LinearPeptide(modified_sequence + f"/{charge}")
        outs = self._convert_linear_peptide(peptide, charge=charge)
        return outs, peptide


def collate_inputs(batch: list[dict[str, np.array]]) -> dict[str, np.array]:
    out_input_ids = np.stack([t["input_ids_s"] for t in batch], axis=0)
    out_position_ids = np.stack([t["position_ids_s"] for t in batch], axis=0)
    out_charge = np.stack([t["charge_1"] for t in batch], axis=0)
    out_src_key_padding_mask = np.stack(
        [t["src_key_padding_mask_s"] for t in batch], axis=0
    )
    return {
        "input_ids_ns": out_input_ids,
        "position_ids_ns": out_position_ids.astype(np.float32),
        "charge_n1": out_charge,
        "src_key_padding_mask_ns": out_src_key_padding_mask,
    }


class OnnxPeptideTransformer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.converter = OnnxSequenceTensorConverter()
        self.session = None

        # session = InferenceSession(model_path)
        # output = session.run(None, {k: v.numpy() for k, v in test_inputs.items()})

    @classmethod
    def default_model(cls) -> "OnnxPeptideTransformer":
        return cls(Path(__file__).parent / "data" / "model.onnx")

    def predict(self, modified_sequence: str, charge: int) -> np.array:
        input_tensors = self.converter.convert(modified_sequence, charge)
        ins = collate_inputs([input_tensors] * 9)

        if self.session is None:
            self.session = InferenceSession(self.model_path)

        output = self.session.run(None, ins)
        return output[0][0]

    def predict_batched(
        self, inputs: Iterator[tuple[str, int]]
    ) -> Generator[np.array, None, None]:
        if self.session is None:
            self.session = InferenceSession(self.model_path)

        # Note 9 is hard-coded bc the model has fixed batch size of 9
        for batch in batched(inputs, n=9):
            ragged = False
            batch_len = len(batch)
            if batch_len != 9:
                ragged = True
                batch = list(batch)
                batch.extend([batch[-1]] * (9 - len(batch)))

            inputs = collate_inputs([self.converter.convert(x, y) for x, y in batch])

            outs = self.session.run(None, inputs)[0]

            if ragged:
                outs = outs[:batch_len]

            yield outs

    def predict_batched_annotated(
        self,
        inputs: Iterator[tuple[str, int]],
        min_intensity: float = 0.001,
        min_ordinal: int = 0,
        max_ordinal: int = 1000,
    ):
        if self.session is None:
            self.session = InferenceSession(self.model_path)

        # Note 9 is hard-coded bc the model has fixed batch size of 9
        for batch in batched(inputs, n=9):
            ragged = False
            batch_len = len(batch)
            if batch_len != 9:
                ragged = True
                batch = list(batch)
                batch.extend([batch[-1]] * (9 - len(batch)))

            converted_withpeps = [
                self.converter.convert_with_peptide(x, y) for x, y in batch
            ]
            peptides = [x[1] for x in converted_withpeps]
            inputs = collate_inputs([x[0] for x in converted_withpeps])
            outs = self.session.run(None, inputs)[0]

            if ragged:
                outs = outs[:batch_len]
                peptides = peptides[:batch_len]

            # self.converter.intensity_tensor_config.tensor_to_elems
            for pep, oue in zip(peptides, outs):
                theo_frags = pep.generate_theoretical_fragments(
                    self.converter.intensity_tensor_config.max_charge,
                    rustyms.FragmentationModel.CidHcd,
                )
                theo_frag_mzs = {
                    (f.ion, f.charge): f.formula.mass() / f.charge
                    for f in theo_frags
                    if f.neutral_loss is None
                }
                back_frags = self.converter.intensity_tensor_config.tensor_to_elems(
                    oue,
                    max_ordinal=min(len(pep.stripped_sequence), max_ordinal),
                    min_ordinal=min_ordinal,
                    min_intensity=min_intensity,
                )
                back_frags = {(f"{k[0]}{k[1]}", k[2]): v for k, v in back_frags.items()}
                out = {}
                for k, v in back_frags.items():
                    if k in theo_frag_mzs:
                        out[f"{k[0]}^{k[1]}"] = (theo_frag_mzs[k], v)

                yield pep, out
