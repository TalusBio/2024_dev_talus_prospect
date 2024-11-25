import re

import polars as pl
from elfragmentador_train.data_utils import MOD_STRIP_REGEX


def test_mod_strip_regex():
    assert (
        re.compile(MOD_STRIP_REGEX).sub("", "[UNIMOD:1]-AC[UNIMOD:4]DEK/2") == "ACDEK/2"
    )


def test_mod_strip_regex_polars():
    series = pl.Series(
        [
            "[UNIMOD:1]-AC[UNIMOD:4]DEK/2",
            "[UNIMOD:1]AC[UNIMOD:4]DEK/2",
            "[UNIMOD:1]AC[UNIMOD:4]DEK[UNIMOD:1]-[UNIMOD:1]/2",
        ],
    )
    out = series.str.replace_all(MOD_STRIP_REGEX, "")
    assert out.to_list() == ["ACDEK/2", "ACDEK/2", "ACDEK/2"]
