import numpy as np

"""
Absolute energy (eV) = (settling NCE) x (Isolation center) / (500 m/z) x (charge factor)

Charge factors are as so:
Charge/charge factor
2/0.9
3/0.85
4/0.8
5/0.75
>5/0.75

So.... if you set the QE to do a NCE of 40 and
you hit a 1,000 m/z ion with a charge state of 1
(40) x (1,000)/500 x 1 = 80 eV
"""

CORR_FACTORS = np.array([float("inf"), 1, 0.9, 0.85, 0.8] + ([0.75] * 5))


def nce_to_ev(nce: float, mz: float, charge: int) -> float:
    """Converts a NCE to an absolute energy.

    This formula is based on the QE formula.

    Parameters
    ----------
        nce (float): The NCE.
        mz (float): The m/z.
        charge (int): The charge.

    Returns
    -------
        float: The absolute energy.

    """
    out = (nce * mz) / (500.0 * CORR_FACTORS[charge])
    return out


#  if you set the QE to do a NCE of 40 and you hit a
#  ... 1,000 m/z ion with a charge state of 1 (40) x (1,000)/500 x 1 = 80 eV
#  so 40 nce on 1000 m/z charge 2 = 88.888 eV)..
