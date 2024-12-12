from dataclasses import dataclass, field
from hashlib import sha256
from typing import Self
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch
from elfragmentador_core.config import IntensityTensorConfig
from elfragmentador_core.converter import SequenceTensorConverter
from elfragmentador_core.nce import nce_to_ev
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from loguru import logger

from .data_utils import MOD_STRIP_REGEX, ef_batch_collate_fn, DatasetSplit
from .utils_extra import simple_timer, batched


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
    # "retention_time",
    # "indexed_retention_time",
    # "andromeda_score",
    # "peptide_length",
    # "base_intensity",
    # "total_intensity",
    "precursor_mz",  # Used to calculate the CE
    # "aligned_collision_energy",  # Used to calculate the CE
    # Used to calculate the CE ... SHOULD use the aligned but TMT_TUM_mod_ubi
    # has nan values ...
    "orig_collision_energy",
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


class TorchSequenceTensorConverter(SequenceTensorConverter):  # noqa: D101
    def convert(
        self,
        row: dict,
        token_cache: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Converts a row of the joint precursor+fragment table to an input batch."""
        if token_cache is None:
            token_cache = {}

        if row["modified_sequence"] in token_cache:
            tokens, positions = token_cache[row["modified_sequence"]]
        else:
            tokens, positions = self.tokenize_proforma(row["modified_sequence"])
            token_cache[row["modified_sequence"]] = (tokens, positions)

        intensity_tensor = self.intensity_tensor_config.elems_to_tensor(
            row["no"],
            row["ion_type"],
            row["charge"],
            row["intensity"],
        )
        charge_ce_tensor = np.array(
            [row["precursor_charge"], row["ce"]], dtype=np.float32
        )
        return {
            "seq_tensor": torch.tensor(tokens).long(),
            "pos_tensor": torch.tensor(positions).float(),
            "charge_ce_tensor": torch.tensor(charge_ce_tensor),
            "intensity_tensor": torch.tensor(intensity_tensor).float(),
        }


def train_val_test_split(  # noqa: D103
    precursors_scanned: pl.LazyFrame,
) -> dict[DatasetSplit, set[str]]:
    uniq_seqs = (
        precursors_scanned.select(
            pl.col("modified_sequence").unique(),
        )
        .with_columns(
            pl.col("modified_sequence")
            .str.replace_all(MOD_STRIP_REGEX, "")
            .alias("stripped_sequence"),
        )
        .collect()
    )

    seq_to_strip = dict(
        zip(
            uniq_seqs["modified_sequence"], uniq_seqs["stripped_sequence"], strict=False
        )
    )
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
class FragmentationDatasetFactory:  # noqa: D101
    fragments_path: str
    precursors_path: str
    partitions_keep: list[str] | None = None
    config: IntensityTensorConfig = field(default_factory=IntensityTensorConfig)
    _split_lazy_frames: dict[DatasetSplit, pl.LazyFrame] | None = None

    def __post_init__(self) -> None:
        hashable = str(self.fragments_path) + str(self.precursors_path)
        if self.partitions_keep is not None:
            partitions_copy = [str(x) for x in self.partitions_keep]
            partitions_copy.sort()
            hashable += "".join(partitions_copy)
        self.hash = sha256(hashable.encode()).hexdigest()
        logger.info(f"Hash: {self.hash}")

    @property
    def split_lazy_frames(self) -> dict[DatasetSplit, pl.LazyFrame]:  # noqa: D102
        if self._split_lazy_frames is None:
            self._split_lazy_frames = self.build_split_combined_frames(
                self.precursors_path,
                self.fragments_path,
            )
        return self._split_lazy_frames

    def get_train_ds(self) -> "FragmentationDataset":  # noqa: D102
        return self.build_split_dataset(DatasetSplit.TRAIN, progress_bar=True)

    def get_val_ds(self) -> "FragmentationDataset":  # noqa: D102
        return self.build_split_dataset(DatasetSplit.VAL, progress_bar=True)

    def get_test_ds(self) -> "FragmentationDataset":  # noqa: D102
        return self.build_split_dataset(DatasetSplit.TEST, progress_bar=True)

    def build_split_dataset(  # noqa: D102
        self,
        split: DatasetSplit,
        progress_bar: bool = False,
    ) -> "FragmentationDataset":
        cache_path = Path("cache") / self.hash  # / f"{split}"
        matches = list(cache_path.glob(f"{split}*.parquet"))

        if matches:
            logger.warning(
                f"Loading from cache: {cache_path}, "
                f" delete the dir if you dont want this ({matches})"
            )
            time.sleep(2)
            return FragmentationDataset.load_from_parquet(matches)

        cache_path.mkdir(parents=True, exist_ok=True)
        tmp = FragmentationDataset.new(
            self.split_lazy_frames[split]["fragments"],
            self.split_lazy_frames[split]["precursors"],
            converter=TorchSequenceTensorConverter(self.config),
            progress_bar=progress_bar,
        )
        cache_prefix = cache_path / f"{split}"
        tmp.save_to_parquet(str(cache_prefix))
        return tmp

    def build_scanned_frames(  # noqa: D102
        self,
        precrsors_path: str,
        fragments_path: str,
    ) -> dict[str, pl.LazyFrame]:
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
                pl.col("partition").is_in(self.partitions_keep),
            )
            precursors = precursors.filter(
                pl.col("partition").is_in(self.partitions_keep),
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

    def build_split_combined_frames(  # noqa: D102
        self,
        precrsors_path: str,
        fragments_path: str,
    ) -> dict[DatasetSplit, pl.LazyFrame]:
        frames = self.build_scanned_frames(precrsors_path, fragments_path)
        split = train_val_test_split(frames["precursors"])

        out = {}
        for k, v in split.items():
            local_precs = frames["precursors"].filter(
                pl.col("modified_sequence").is_in(v),
            )
            local_frags = frames["fragments"].filter(
                pl.col("peptide_sequence").is_in(v),
            )
            out[k] = {
                "fragments": local_frags,
                "precursors": local_precs,
            }

        return out


class FragmentationDataset(Dataset):  # noqa: D101
    def __init__(self, elems: list[dict[str, torch.Tensor]]) -> None:
        self.elems = elems

    @classmethod
    def new(  # noqa: D102
        cls,
        fragments: pl.LazyFrame,
        precursors: pl.LazyFrame,
        converter: TorchSequenceTensorConverter,
        progress_bar: bool = False,
    ) -> Self:
        elems = []
        with simple_timer("Collecting precursor rows"):
            collected_precs = precursors.collect(streaming=True)
            uniq_partitions = collected_precs["partition"].unique()

        col_rename_dict = dict(
            zip(JOIN_FRAGMENT_COLS, JOIN_PRECURSOR_COLS, strict=False)
        )

        for part in tqdm(
            uniq_partitions,
            desc="Collecting fragment rows (per partition)",
            disable=not progress_bar,
        ):
            with simple_timer(f"Joining fragments and precursors ({part})"):
                local_precs = collected_precs.filter(
                    pl.col("partition").is_in([part]),
                ).lazy()
                collected_frags = fragments.filter(
                    pl.col("partition").is_in([part]),
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

                # The TMT_TUM_mod_ubi partition has no aligned energies ...
                ce = nce_to_ev(
                    joint["orig_collision_energy"].to_numpy(),
                    joint["precursor_mz"].to_numpy(),
                    joint["precursor_charge"].to_numpy(),
                )

                joint = joint.with_columns(
                    ce=pl.Series(ce, dtype=pl.Float32),
                )

                missing_ce = joint.filter(
                    pl.col("ce").is_null() | pl.col("ce").is_nan()
                )
                if not missing_ce.is_empty():
                    # TMT_TUM_mod_ubi has no aligned collision energies
                    # So this is needed to make sure I dont accidentally
                    # add that partition ...
                    logger.warning(f"Missing CE: {missing_ce}")
                    raise RuntimeError("Missing CE")

            # TODO: reimplement the cache as an LRU cache in
            # SequenceTensorConverter.tokenize_proforma
            # TODO: Check if the intensity tensor can be built faster
            #       using list -> array build.
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
                    ),
                ):
                    local_elems[row_idx] = converter.convert(
                        row,
                        token_cache=token_cache,
                    )

                assert not any(x is None for x in local_elems)
                elems.extend(local_elems)

        return cls(elems=elems)

    def save_to_parquet(self, path_prefix: str) -> None:
        logger.info(f"Saving to parquet: {path_prefix}")
        for bi, batch in enumerate(
            batched(tqdm(self.elems, desc="Converting to parquet"), 200_000)
        ):
            elems_save = []
            for elem in batch:
                tmp = {k: v.cpu().numpy().tolist() for k, v in elem.items()}
                elems_save.append(tmp)

            curr_path = path_prefix + f"{bi}.parquet"
            logger.info(f"Saving to parquet chunk {bi}: {curr_path}")
            pl.DataFrame(elems_save).write_parquet(curr_path)

    @classmethod
    def load_from_parquet(cls, paths: list[str]) -> "FragmentationDataset":
        logger.info(f"Loading from parquet: {paths}")
        elems_use = []
        for path in tqdm(paths):
            elems = pl.read_parquet(path)
            for elem in elems.iter_rows(named=True):
                elem_o = {k: torch.Tensor(v) for k, v in elem.items()}
                elem_o["seq_tensor"] = elem_o["seq_tensor"].long()
                elem_o["pos_tensor"] = elem_o["pos_tensor"].float()
                elem_o["charge_ce_tensor"] = elem_o["charge_ce_tensor"].float()
                elem_o["intensity_tensor"] = elem_o["intensity_tensor"].float()
                elems_use.append(elem_o)
        return cls(elems=elems_use)

    def __len__(self) -> int:  # noqa: ANN201, D102
        return len(self.elems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: ANN201, D102
        return self.elems[idx]

    def with_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:  # noqa: ANN201, D102
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=ef_batch_collate_fn,
        )


def _testing_row() -> dict:
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
        "ce": 88.8,
    }
