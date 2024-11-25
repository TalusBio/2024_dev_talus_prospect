import numpy as np

SPC_ORD = ord(" ")


def make_src_key_padding_mask(
    input_ids_ns: np.array, pad_token_id: int = SPC_ORD
) -> np.array:
    """Makes a mask for the src key padding.

    Args:
        input_ids_ns (np.array): A tensor of shape (N, S) where N is the
            batch size and S is the sequence length.
        pad_token_id (int): The token id to use for padding.

    Returns
    -------
        np.array: A tensor of shape (N, S) where N is the batch size
            and S is the sequence length.
            The values are either 0 or -inf.

    """
    # input_ids_ns: [batch_size, seq_length]
    mask_ns = np.zeros_like(input_ids_ns, dtype=np.float32)
    mask_ns[input_ids_ns == pad_token_id] = -np.inf
    return mask_ns
