import numpy as np
from elfragmentador_train.fragment_dataset import (
    TorchSequenceTensorConverter,
    _testing_row,
    ef_batch_collate_fn,
)

# factory = FragmentationDatasetFactory(
#     fragments_path="part_data/fragments_pq", precursors_path="part_data/precursors_pq"
# )
#
# train_ds = factory.get_train_ds()
# train_ds[0]
#
# train_dl = train_ds.with_dataloader(batch_size=32, shuffle=True)
# for batch in train_dl:
#     for k, t in batch.items():
#         print(k)
#         print(t.shape)
#     break


def test_smoke_convertsion():
    conv = TorchSequenceTensorConverter()
    sample_row = _testing_row()
    sample_row2 = _testing_row()
    sample_row2["precursor_charge"] = 3
    sample_row2["modified_sequence"] += "K"
    tensors = conv.convert(sample_row)
    tensors2 = conv.convert(sample_row2)
    _ef_batch = ef_batch_collate_fn([tensors, tensors2])

    conv = TorchSequenceTensorConverter()
    out1 = conv.tokenize_proforma("[UNIMOD:1]-AC[UNIMOD:4]DEK/2")

    # Test that it handles the wrong notation in prospect correctly
    out2 = conv.tokenize_proforma("[UNIMOD:1]AC[UNIMOD:4]DEK/2")
    assert np.all(out1[0] == out2[0])
    assert np.all(out1[1] == out2[1])
