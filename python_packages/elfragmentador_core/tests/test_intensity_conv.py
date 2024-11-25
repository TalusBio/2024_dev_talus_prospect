from elfragmentador_core.config import IntensityTensorConfig


def _test_arrs():
    out = {
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
    }

    return out


def test_series_tuple():
    config = IntensityTensorConfig(max_charge=2, ion_types=["b", "y"])

    ind = config.series_indices(2, "b", 1)
    assert ind == (2, 0)
    ind = config.series_indices(2, "y", 1)
    assert ind == (2, 1)
    ind = config.series_indices(2, "b", 2)
    assert ind == (2, 2)
    ind = config.series_indices(2, "y", 2)
    assert ind == (2, 3)

    bw = config.series_tuple(2, 0)
    assert bw == ("b", 2, 1)
    bw = config.series_tuple(2, 1)
    assert bw == ("y", 2, 1)
    bw = config.series_tuple(2, 2)
    assert bw == ("b", 2, 2)
    bw = config.series_tuple(2, 3)
    assert bw == ("y", 2, 2)

    for ordinal in [2, 4, 6, 8]:
        for charge in [1, 2]:
            for itype in ["b", "y"]:
                fw = config.series_indices(ordinal, itype, charge)
                bw = config.series_tuple(fw[0], fw[1])
                assert bw == (
                    itype,
                    ordinal,
                    charge,
                ), f"{fw} vs {bw}, ordinal: {ordinal}, charge: {charge}, itype: {itype}"


def test_intensity_conv():
    config = IntensityTensorConfig(max_charge=2, ion_types=["b", "y"])
    tensor = config.elems_to_tensor(
        ion_ordinals=[
            1,
            2,
            3,
            4,
        ],
        ion_types=["b", "b", "b", "y"],
        ion_charges=[1, 2, 1, 2],
        intensities=[0.1, 0.2, 0.3, 0.4],
    )

    back_to_elems = config.tensor_to_elems(tensor)
    int_back_to_elems = {k: int(v * 10) for k, v in back_to_elems.items()}
    assert int_back_to_elems == {
        ("b", 1, 1): 1,
        ("b", 2, 2): 2,
        ("b", 3, 1): 3,
        ("y", 4, 2): 4,
    }
