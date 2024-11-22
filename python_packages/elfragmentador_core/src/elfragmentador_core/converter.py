from .config import IntensityTensorConfig
from dataclasses import dataclass, field
import rustyms
import numpy as np
from functools import lru_cache


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

    def tokenize_proforma(
        self, proforma: str, padded_length: int | None = None
    ) -> tuple[np.array, np.array]:
        """Tokenizes a proforma peptide.

        Args:
            proforma (str): The proforma peptide to tokenize.
            padded_length (int | None, optional): The length of the padded sequence. Defaults to None.

        Returns:
            tuple[np.array, np.array]: A tuple of two numpy arrays. The first contains the token values
            and the second contains the token positions.

        Raises:
            RuntimeError: If the padding length is smaller than the length of the tokenized sequence.
        """

        # Prospect has wrong proforma peptides ... [UNIMOD:1]K should be [UNIMOD:1]-K
        # So ... if it starts with "[" the first character after the first "]" should be "-"
        proforma = _fix_prospect_proforma(proforma)
        peptide = rustyms.LinearPeptide(proforma)
        return self.tokenize_linear_peptide(peptide, padded_length)

    def tokenize_linear_peptide(
        self, peptide: rustyms.LinearPeptide, padded_length: int | None = None
    ) -> tuple[list[int], list[int]]:
        mod_tokens = self.tokenize_mods(peptide)
        seq_tokens = self.tokenize_stripped_sequence(peptide.stripped_sequence)
        final_tokenized = seq_tokens + mod_tokens
        if padded_length is not None:
            add_len = padded_length - len(final_tokenized)
            if add_len > 0:
                final_tokenized = (
                    final_tokenized + [(len(final_tokenized) + 1, ord(" "))] * add_len
                )
            elif add_len < 0:
                msg = "Padded length is smaller than the base sequence"
                msg += f"{padded_length} vs {len(final_tokenized)}"
                msg += f"proforma: {peptide}"
                raise RuntimeError(msg)

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
