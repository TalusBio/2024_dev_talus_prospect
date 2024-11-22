"""
Expected usage of the dataset:

```
factory = FragmentationDatasetFactory(
    fragments_path="part_data/fragments_pq", precursors_path="part_data/precursors_pq"
)

train_ds = factory.get_train_ds() # This gets the dataset

# This makes it a dataloader with the right collate function that pads the tensors
train_dl = train_ds.with_dataloader(batch_size=32, shuffle=True)
for batch in train_dl:
    for k, t in batch.items():
        print(k)
        print(t.shape)
    break
"""

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import polars as pl
import numpy as np
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
import rustyms
import enum
from typing import Self
from .data_utils import ef_batch_collate_fn, MOD_STRIP_REGEX
from .utils_extra import simple_timer


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


FRAGMENT_COLS = [
    # Used for ID
    "ion_type",
    "no",
    "charge",
    "intensity",
    # Unused
    # "experimental_mass",
    # "theoretical_mass",
    # "__index_level_0__",
    # Used for filtering
    "neutral_loss",
    "fragment_score",
    # Used to join
    "peptide_sequence",
    "scan_number",
    "raw_file",
    "partition",
]
PRECURSOR_COLS = [
    # "precursor_intensity",
    # "mz",
    # "precursor_mz",
    # "retention_time",
    # "indexed_retention_time",
    # "andromeda_score",
    # "peptide_length",
    # "base_intensity",
    # "total_intensity",
    # "orig_collision_energy",
    # "aligned_collision_energy",
    # Used for filtering
    "mass_analyzer",
    "fragmentation",
    # Used for ID
    "precursor_charge",
    # Used to join
    "scan_number",
    "modified_sequence",
    "raw_file",
    "partition",
]
JOIN_PRECURSOR_COLS = ["modified_sequence", "scan_number", "raw_file", "partition"]
JOIN_FRAGMENT_COLS = ["peptide_sequence", "scan_number", "raw_file", "partition"]


def _fix_prospect_proforma(proforma: str) -> str:
    if proforma.startswith("[") and proforma[proforma.index("]") + 1] != "-":
        first_mod_partition = proforma.index("]")
        start_len = len(proforma)
        proforma = (
            proforma[: first_mod_partition + 1]
            + "-"
            + proforma[first_mod_partition + 1 :]
        )
        end_len = len(proforma)
        assert (end_len - start_len) == 1
    return proforma


@dataclass(frozen=True, slots=True, eq=True)
class IntensityTensorConfig:
    max_charge: int = 2
    ion_types: tuple[str] = ("b", "y")

    def series_indices(
        self, ion_ordinal: int, ion_type: str, charge: int
    ) -> tuple[int, int] | None:
        if (charge > self.max_charge) or (charge <= 0):
            raise RuntimeError(
                f"Invalid inputs for series_indices: {ion_ordinal}, {ion_type}, {charge}"
            )
        jj = self.ion_types.index(ion_type) * (charge)
        ii = ion_ordinal
        return (ii, jj)

    def build_empty_tensor(self, max_ordinal: int) -> np.array:
        return np.zeros(
            (max_ordinal, len(self.ion_types) * self.max_charge), dtype=np.float32
        )

    def elems_to_tensor(
        self,
        ion_ordinals: list[int],
        ion_types: list[str],
        ion_charges: list[int],
        intensities: list[float],
    ) -> np.array:
        max_ordinal = max(ion_ordinals)
        tensor = self.build_empty_tensor(max_ordinal + 1)
        for ion_ordinal, ion_type, ion_charge, intensity in zip(
            ion_ordinals, ion_types, ion_charges, intensities, strict=True
        ):
            indices = self.series_indices(ion_ordinal, ion_type, ion_charge)
            tensor[indices] = intensity
        return tensor


@dataclass(frozen=True, slots=True, eq=True)
class SequenceTensorConverter:
    intensity_tensor_config: IntensityTensorConfig = field(
        default_factory=IntensityTensorConfig
    )

    def tokenize_mods(
        self, peptide: rustyms.LinearPeptide
    ) -> tuple[np.array, np.array]:
        MOD_TOKENS = self.mod_tokens()
        mods: list[tuple[int, str]] = []
        for i, mod in enumerate(peptide.sequence):
            if mod.modifications:
                if len(mod.modifications) > 1:
                    raise RuntimeError(
                        f"Peptide {peptide.stripped_sequence} has multiple modifications at position {i}"
                    )
                mods.append((i, str(mod.modifications[0])))

        if peptide.c_term is not None:
            mods.append((len(peptide.sequence), str(peptide.c_term)))

        if peptide.n_term is not None:
            mods.append((0, str(peptide.n_term)))

        mod_tokens = [(mod[0], MOD_TOKENS[mod[1]]) for mod in mods]
        return mod_tokens

    def tokenize_stripped_sequence(
        self, stripped_sequence: str
    ) -> tuple[np.array, np.array]:
        seq_string = "^" + stripped_sequence + "$"
        seq_tokens = [(i, ord(c)) for i, c in enumerate(seq_string)]
        return seq_tokens

    def tokenize_proforma(self, proforma: str) -> tuple[np.array, np.array]:
        # Prospect has wrong proforma peptides ... [UNIMOD:1]K should be [UNIMOD:1]-K
        # So ... if it starts with "[" the first character after the first "]" should be "-"
        proforma = _fix_prospect_proforma(proforma)
        peptide = rustyms.LinearPeptide(proforma)
        mod_tokens = self.tokenize_mods(peptide)
        seq_tokens = self.tokenize_stripped_sequence(peptide.stripped_sequence)
        final_tokenized = seq_tokens + mod_tokens

        positions_array = np.array([x[0] for x in final_tokenized])
        tokens_array = np.array([x[1] for x in final_tokenized])
        # One assertion a day keeps hidden bugs away
        assert len(positions_array) == len(tokens_array)
        return tokens_array, positions_array

    @staticmethod
    @lru_cache(maxsize=1)
    def mod_tokens() -> dict[str, int]:
        """

        ```python
        modnames = []
        mods = [
            "C[UNIMOD:4]",
            "E[UNIMOD:27]",
            "K[UNIMOD:121]",
            "K[UNIMOD:122]",
            "K[UNIMOD:1289]",
            "K[UNIMOD:1363]",
            "K[UNIMOD:1848]",
            "K[UNIMOD:1849]",
            "K[UNIMOD:1]",
            "K[UNIMOD:34]",
            "K[UNIMOD:36]",
            "K[UNIMOD:37]",
            "K[UNIMOD:58]",
            "K[UNIMOD:64]",
            "K[UNIMOD:737]",
            "K[UNIMOD:747]",
            "M[UNIMOD:35]",
            "P[UNIMOD:35]",
            "Q[UNIMOD:28]",
            "R[UNIMOD:34]",
            "R[UNIMOD:36]",
            "R[UNIMOD:7]",
            "S[UNIMOD:21]",
            "S[UNIMOD:43]",
            "T[UNIMOD:21]",
            "T[UNIMOD:43]",
            "Y[UNIMOD:21]",
            "[UNIMOD:1]-K",
            "[UNIMOD:737]-K",
        ]

        for mod in mods:
            pep = rustyms.LinearPeptide(mod)
            local_mods = [str(x.modifications[0]) for x in pep.sequence if x.modifications]
            modnames.extend(local_mods)
            if pep.n_term is not None:
                modnames.append(str(pep.n_term))
            if pep.c_term is not None:
                modnames.append(str(pep.c_term))

        unique_modnames = list(set(modnames))
        unique_modnames.sort()

        # 32 and from 65 to 122 are taken
        # 200 seems to be mostly accentuated versions of the basic a-zA-Z chars
        {k: i + 200 for i, k in enumerate(unique_modnames)}
        ```

        """
        MOD_TOKENS = {
            "U:Acetyl": 200,
            "U:Butyryl": 201,
            "U:Carbamidomethyl": 202,
            "U:Crotonyl": 203,
            "U:Deamidated": 204,
            "U:Dimethyl": 205,
            "U:Formyl": 206,
            "U:GG": 207,
            "U:Gln->pyro-Glu": 208,
            "U:Glu->pyro-Glu": 209,
            "U:Gluratylation": 210,
            "U:HexNAc": 211,
            "U:Malonyl": 212,
            "U:Methyl": 213,
            "U:Oxidation": 214,
            "U:Phospho": 215,
            "U:Propionyl": 216,
            "U:Succinyl": 217,
            "U:TMT6plex": 218,
            "U:Trimethyl": 219,
            "U:hydroxyisobutyryl": 220,
        }
        return MOD_TOKENS

    def convert(
        self, row: dict, token_cache: dict | None = None
    ) -> dict[str, torch.Tensor]:
        if token_cache is None:
            token_cache = {}

        if row["modified_sequence"] in token_cache:
            tokens, positions = token_cache[row["modified_sequence"]]
        else:
            tokens, positions = self.tokenize_proforma(row["modified_sequence"])
            token_cache[row["modified_sequence"]] = (tokens, positions)

        intensity_tensor = self.intensity_tensor_config.elems_to_tensor(
            row["no"], row["ion_type"], row["charge"], row["intensity"]
        )
        charge_tensor = np.array([row["precursor_charge"]], dtype=np.float32)
        return {
            "seq_tensor": torch.tensor(tokens).long(),
            "pos_tensor": torch.tensor(positions).float(),
            "charge_tensor": torch.tensor(charge_tensor),
            "intensity_tensor": torch.tensor(intensity_tensor).float(),
        }


def train_val_test_split(
    precursors_scanned: pl.LazyFrame,
) -> dict[DatasetSplit, set[str]]:
    print("Building train-test-val splits")

    uniq_seqs = (
        precursors_scanned.select(
            pl.col("modified_sequence").unique(),
        )
        .with_columns(
            pl.col("modified_sequence")
            .str.replace_all(MOD_STRIP_REGEX, "")
            .alias("stripped_sequence")
        )
        .collect()
    )

    seq_to_strip = {
        m: s
        for m, s in zip(uniq_seqs["modified_sequence"], uniq_seqs["stripped_sequence"])
    }
    uniq_strip = list(set(seq_to_strip.values()))
    uniq_strip.sort()

    # to a 50/40/10 split
    rng = np.random.default_rng(42)
    rng.shuffle(uniq_strip)

    train_test_part_point = int(0.7 * len(uniq_strip))
    train_val_part_point = int(0.5 * train_test_part_point)

    train_strip = set(uniq_strip[:train_val_part_point])
    val_strip = set(uniq_strip[train_val_part_point:train_test_part_point])
    test_strip = set(uniq_strip[train_test_part_point:])

    train_seqs = {k for k, v in seq_to_strip.items() if v in train_strip}
    val_seqs = {k for k, v in seq_to_strip.items() if v in val_strip}
    test_seqs = {k for k, v in seq_to_strip.items() if v in test_strip}

    return {
        DatasetSplit.TRAIN: train_seqs,
        DatasetSplit.VAL: val_seqs,
        DatasetSplit.TEST: test_seqs,
    }


@dataclass
class FragmentationDatasetFactory:
    fragments_path: str
    precursors_path: str
    partitions_keep: list[str] | None = None
    config: IntensityTensorConfig = field(default_factory=IntensityTensorConfig)
    _split_lazy_frames: dict[DatasetSplit, pl.LazyFrame] | None = None

    @property
    def split_lazy_frames(self) -> dict[DatasetSplit, pl.LazyFrame]:
        if self._split_lazy_frames is None:
            self._split_lazy_frames = self.build_split_combined_frames(
                self.precursors_path, self.fragments_path
            )
        return self._split_lazy_frames

    def get_train_ds(self) -> "FragmentationDataset":
        return self.build_split_dataset(DatasetSplit.TRAIN, progress_bar=True)

    def get_val_ds(self) -> "FragmentationDataset":
        return self.build_split_dataset(DatasetSplit.VAL, progress_bar=True)

    def get_test_ds(self) -> "FragmentationDataset":
        return self.build_split_dataset(DatasetSplit.TEST, progress_bar=True)

    def build_split_dataset(
        self, split: DatasetSplit, progress_bar: bool = False
    ) -> "FragmentationDataset":
        print(f"Building dataset for {split}")
        return FragmentationDataset.new(
            self.split_lazy_frames[split]["fragments"],
            self.split_lazy_frames[split]["precursors"],
            converter=SequenceTensorConverter(self.config),
            progress_bar=progress_bar,
        )

    def build_scanned_frames(
        self, precrsors_path: str, fragments_path: str
    ) -> dict[str, pl.LazyFrame]:
        print(f"Building frames for {fragments_path} and {precrsors_path}")
        fragments = (
            pl.scan_parquet(fragments_path)
            .filter(pl.col("neutral_loss") == "", pl.col("fragment_score") > 0.5)
            .filter(pl.col("ion_type").is_in(self.config.ion_types))
            .filter(pl.col("charge") <= self.config.max_charge)
        )
        precursors = (
            pl.scan_parquet(precrsors_path)
            .filter(pl.col("mass_analyzer") == "FTMS", pl.col("fragmentation") == "HCD")
            .select(PRECURSOR_COLS)
        )

        if self.partitions_keep is not None:
            fragments = fragments.filter(
                pl.col("partition").is_in(self.partitions_keep)
            )
            precursors = precursors.filter(
                pl.col("partition").is_in(self.partitions_keep)
            )

        fragments = (
            fragments.select(FRAGMENT_COLS)
            .group_by(JOIN_FRAGMENT_COLS)
            .agg(pl.all().exclude(JOIN_FRAGMENT_COLS))
            .filter(pl.col("intensity").list.len() > 3)
        )

        return {
            "fragments": fragments,
            "precursors": precursors,
        }

    def build_split_combined_frames(
        self, precrsors_path: str, fragments_path: str
    ) -> dict[DatasetSplit, pl.LazyFrame]:
        frames = self.build_scanned_frames(precrsors_path, fragments_path)
        split = train_val_test_split(frames["precursors"])

        print("Pre-building splits")
        out = {}
        for k, v in split.items():
            local_precs = frames["precursors"].filter(
                pl.col("modified_sequence").is_in(v)
            )
            local_frags = frames["fragments"].filter(
                pl.col("peptide_sequence").is_in(v)
            )
            out[k] = {
                "fragments": local_frags,
                "precursors": local_precs,
            }

            # local_frags.join(
            #     local_precs,
            #     left_on=JOIN_FRAGMENT_COLS,
            #     right_on=JOIN_PRECURSOR_COLS,
            #     how="inner",
            #     # validate="1:1",
            #     # For some reason this is not passing ...
            # )
        return out


class FragmentationDataset(Dataset):
    def __init__(self, elems: list[dict[str, torch.Tensor]]):
        self.elems = elems

    @classmethod
    def new(
        cls,
        fragments: pl.LazyFrame,
        precursors: pl.LazyFrame,
        converter: SequenceTensorConverter,
        progress_bar: bool = False,
    ) -> Self:
        elems = []
        with simple_timer("Collecting precursor rows"):
            collected_precs = precursors.collect(streaming=True)
            uniq_partitions = collected_precs["partition"].unique()

        col_rename_dict = {
            k: v for k, v in zip(JOIN_FRAGMENT_COLS, JOIN_PRECURSOR_COLS)
        }

        for part in tqdm(
            uniq_partitions,
            desc="Collecting fragment rows (per partition)",
            disable=not progress_bar,
        ):
            with simple_timer(f"Joining fragments and precursors ({part})"):
                local_precs = collected_precs.filter(
                    pl.col("partition").is_in([part])
                ).lazy()
                collected_frags = fragments.filter(
                    pl.col("partition").is_in([part]),
                    # These were needed due to memory issues
                    # pl.col("raw_file").is_in(local_precs["raw_file"].unique()),
                    # pl.col("scan_number").is_in(local_precs["scan_number"].unique()),
                ).rename(col_rename_dict)

                # AS OF 2024-11-20
                # Having this separate join is needed due to a bug in polars.
                # https://github.com/pola-rs/polars/issues/19822

                # joint = collected_frags.join(
                #     local_precs,
                #     left_on=JOIN_FRAGMENT_COLS,
                #     right_on=JOIN_PRECURSOR_COLS,
                #     how="inner",
                # )
                joint = collected_frags.join(
                    local_precs,
                    on=JOIN_PRECURSOR_COLS,
                    how="inner",
                ).collect(streaming=True)

            print(joint)

            # TODO: reimplement the cache as an LRU cache in SequenceTensorConverter.tokenize_proforma
            # TODO: Check if the intensity tensor can be built faster using list -> array build.
            #       Since in-place mod COULD be faster, but it's not clear if it is.
            with simple_timer(f"Converting rows ({part})"):
                token_cache = {}
                local_elems = [None] * len(joint)
                for row_idx, row in enumerate(
                    tqdm(
                        joint.iter_rows(named=True),
                        total=len(joint),
                        disable=not progress_bar,
                        desc="Converting rows",
                    )
                ):
                    local_elems[row_idx] = converter.convert(
                        row, token_cache=token_cache
                    )

                assert not any(x is None for x in local_elems)
                elems.extend(local_elems)

        return cls(elems=elems)

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.elems[idx]

    def with_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=ef_batch_collate_fn
        )


def _testing_row():
    return {
        "modified_sequence": "AANDAGYFNDEM[UNIMOD:35]APIEVK[UNIMOD:121]TK",
        "scan_number": 26983,
        "raw_file": "02330a_GD3_3991_09_PTM_TrainKit_Kmod_Ubiquitinyl_200fmol_3xHCD_R1",
        "partition": "Kmod_GlyGly",
        "ion_type": [
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
        ],
        "no": [
            4,
            5,
            7,
            8,
            9,
            10,
            19,
            11,
            12,
            13,
            15,
            2,
            3,
            4,
            5,
            7,
            15,
            16,
            8,
            18,
            19,
            9,
            10,
            11,
            12,
            13,
            15,
        ],
        "charge": [
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "intensity": [
            0.07,
            0.04,
            0.19,
            0.19,
            0.05,
            0.05,
            0.05,
            0.06,
            0.09,
            0.22,
            0.07,
            0.17,
            0.57,
            0.27,
            0.29,
            1.0,
            0.27,
            0.43,
            0.31,
            0.29,
            0.05,
            0.17,
            0.24,
            0.25,
            0.48,
            0.44,
            0.07,
        ],
        "mass_analyzer": "FTMS",
        "fragmentation": "HCD",
        "precursor_charge": 2,
    }
