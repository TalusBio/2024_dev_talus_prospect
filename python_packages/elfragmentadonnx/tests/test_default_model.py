from elfragmentadonnx.model import OnnxPeptideTransformer
import random
import rustyms


def random_peptide_builder(length_range=(5, 10), num_peptides=10):
    aa_candidate = [
        "A",
        "C",
        "C[UNIMOD:4]",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "M[UNIMOD:35]",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    peptides = []
    for _ in range(num_peptides):
        length = random.randint(*length_range)
        peptide = "".join(random.choice(aa_candidate) for _ in range(length))
        peptides.append(peptide)
    return peptides


def random_charged_peptide_builder(
    length_range=(5, 10), num_peptides=10, charge_range=(2, 3)
):
    peptides = random_peptide_builder(length_range, num_peptides)
    charged_peptides = []
    for peptide in peptides:
        charge = random.randint(*charge_range)
        charged_peptides.append((peptide, charge))
    return charged_peptides


def test_smoke():
    model = OnnxPeptideTransformer.default_model()
    outs = model.predict("MYPEPTIDEK", 2)

    assert outs.shape == (30, 4)

    peptides = random_charged_peptide_builder(num_peptides=1000)
    first = True
    for x in model.predict_batched(peptides):
        assert len(x.shape) == 3
        assert x.shape[-1] == 4

        if first:
            assert x.shape[0] == 9
            assert x.shape == (9, 30, 4)
            first = False

    for x in model.predict_batched_annotated(peptides):
        assert len(x) == 2
        assert isinstance(x[0], rustyms.LinearPeptide)
        assert isinstance(x[1], dict)
