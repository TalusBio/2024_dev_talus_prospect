from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=True)
class IntensityTensorConfig:
    """Configuration for how the intensity tensor is built."""

    max_charge: int = 2
    ion_types: tuple[str] = ("b", "y")

    def series_indices(
        self,
        ion_ordinal: int,
        ion_type: str,
        charge: int,
    ) -> tuple[int, int] | None:
        """Returns the indices for a given series, ordinal and charge.

        Parameters
        ----------
            ion_ordinal (int): The ion ordinal.
            ion_type (str): The ion type.
            charge (int): The ion charge.

        Returns
        -------
            tuple[int, int] | None: The indices or None if the charge is invalid.

        Raises
        ------
            RuntimeError: If the charge is greater than the max charge allowed.
            IndexError: If the ion series is not supported.

        """
        if (charge > self.max_charge) or (charge <= 0):
            raise RuntimeError(
                "Invalid inputs for series_indices:"
                f" {ion_ordinal}, {ion_type}, {charge}",
            )
        jj = self.ion_types.index(ion_type) + ((charge - 1) * len(self.ion_types))
        ii = ion_ordinal
        return (ii, jj)

    def series_tuple(self, ii: int, jj: int) -> tuple[str, int, int]:
        """Returns the series, ordinal and charge for a given index.

        Parameters
        ----------
            ii (int): The ion ordinal.
            jj (int): The ion index.

        Returns
        -------
            tuple[str, int, int]: The series, ordinal and charge.

        """
        series = self.ion_types[jj % self.max_charge]
        ordinal = ii
        charge = (jj // self.max_charge) + 1  # charge is 1-indexed
        return (series, ordinal, charge)

    def build_empty_tensor(self, max_ordinal: int) -> np.array:
        """Builds an empty tensor of the given size.

        The tensor is initialized to zeros.
        And the size is (max_ordinal, max_charge * len(ion_types)).

        Parameters
        ----------
            max_ordinal (int): The maximum ordinal.

        Returns
        -------
            np.array: The empty tensor.

        """
        return np.zeros(
            (max_ordinal, len(self.ion_types) * self.max_charge),
            dtype=np.float32,
        )

    def elems_to_tensor(
        self,
        ion_ordinals: list[int],
        ion_types: list[str],
        ion_charges: list[int],
        intensities: list[float],
    ) -> np.array:
        """Converts a list of ion ordinals, types, charges and intensities to a tensor.

        Parameters
        ----------
            ion_ordinals (list[int]): The ion ordinals.
            ion_types (list[str]): The ion types.
            ion_charges (list[int]): The ion charges.
            intensities (list[float]): The intensities.

        Returns
        -------
            np.array: The tensor.

        Raises
        ------
            RuntimeError: If the charge is greater than the max charge allowed.
            IndexError: If the ion seties is not supported.

        """
        max_ordinal = max(ion_ordinals)
        tensor = self.build_empty_tensor(max_ordinal + 1)
        for ion_ordinal, ion_type, ion_charge, intensity in zip(
            ion_ordinals,
            ion_types,
            ion_charges,
            intensities,
            strict=True,
        ):
            indices = self.series_indices(ion_ordinal, ion_type, ion_charge)
            tensor[indices] = intensity
        return tensor

    def tensor_to_elems(
        self,
        tensor: np.array,
        min_intensity: float = 0.001,
        min_ordinal: int = 0,
        max_ordinal: int = 1000,
    ) -> dict[tuple[str, int, int], float]:
        """Converts a tensor to a dictionary of elements.

        The dictionary is keyed by the series, ordinal and charge.
        The values are the intensity.

        Parameters
        ----------
            tensor (np.array): The tensor.
            min_intensity (float, optional): The minimum intensity. Defaults to 0.001.
            min_ordinal (int, optional): The minimum ordinal. Defaults to 0.
            max_ordinal (int, optional): The maximum ordinal. Defaults to 1000.

        Returns
        -------
            dict[tuple[str, int, int], float]: The dictionary of elements.

        """
        out = {}
        for ii in range(min_ordinal, tensor.shape[0]):
            for jj in range(tensor.shape[1]):
                if (local_int := tensor[ii, jj]) > min_intensity:
                    index = self.series_tuple(ii, jj)
                    if index[1] < max_ordinal:
                        assert index not in out, f"Duplicate index: {index}"
                        out[index] = local_int.item()

        return out
