
from dataclasses import dataclass
import numpy as np

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
    
    def series_tuple(self, ii, jj) -> tuple[str, int,  int]:
        series = self.ion_types[jj // self.max_charge]
        ordinal = ii
        charge = jj % self.max_charge
        return (series, ordinal, charge)

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
