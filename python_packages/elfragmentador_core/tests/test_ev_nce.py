from elfragmentador_core.nce import nce_to_ev
import numpy as np


def test_nce_to_ev():
    assert nce_to_ev(40, 1000, 1) == 80
    assert np.allclose(nce_to_ev(40, 1000, 2), 88.888888888)


def test_nce_to_ev_numpy():
    mz = np.array([1000, 1000])
    nce = np.array([40, 40])
    charge = np.array([1, 2])
    out = nce_to_ev(nce, mz, charge)
    assert np.allclose(out, np.array([80, 88.888]))
