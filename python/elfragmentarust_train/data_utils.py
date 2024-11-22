import torch

MOD_STRIP_REGEX = r"(-)?\[.*?\](-)?"


def batch_to_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # input_ids_ns, position_ids_ns, src_key_padding_mask_ns, charge_n1
    input_ids_ns = batch["seq_tensor"]
    position_ids_ns = batch["pos_tensor"]
    charge_n1 = batch["charge_tensor"]
    src_key_padding_mask_ns = make_src_key_padding_mask(
        input_ids_ns, pad_token_id=ord(" ")
    )
    return {
        "input_ids_ns": input_ids_ns,
        "position_ids_ns": position_ids_ns,
        "src_key_padding_mask_ns": src_key_padding_mask_ns,
        "charge_n1": charge_n1,
    }


def collate_tf_inputs(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    out_input_ids = torch.nn.utils.rnn.pad_sequence(
        [t["input_ids_ns"] for t in batch], batch_first=True, padding_value=ord(" ")
    )
    out_position_ids = torch.nn.utils.rnn.pad_sequence(
        [t["position_ids_ns"] for t in batch], batch_first=True, padding_value=-1
    )
    out_charge = torch.stack([t["charge_n1"] for t in batch])
    out_src_key_padding_mask = torch.nn.utils.rnn.pad_sequence(
        [t["src_key_padding_mask_ns"] for t in batch],
        batch_first=True,
        padding_value=-torch.inf,
    )

    return {
        "input_ids_ns": out_input_ids,
        "position_ids_ns": out_position_ids,
        "charge_n1": out_charge,
        "src_key_padding_mask_ns": out_src_key_padding_mask,
    }


def make_src_key_padding_mask(input_ids_ns, pad_token_id=0):
    """Makes a mask for the src key padding.

    Args:
        input_ids_ns (torch.Tensor): A tensor of shape (N, S) where N is the
            batch size and S is the sequence length.
        pad_token_id (int): The token id to use for padding.
    Returns:
        torch.Tensor: A tensor of shape (N, S) where N is the batch size
            and S is the sequence length.
            The values are either 0 or -inf.
    """

    # input_ids_ns: [batch_size, seq_length]
    mask_ns = torch.zeros_like(input_ids_ns, dtype=torch.float)
    mask_ns[input_ids_ns == pad_token_id] = -torch.inf
    return mask_ns


def ef_batch_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    # Pad to size the seqs, positoins and intensity tensors

    out_seqs = torch.nn.utils.rnn.pad_sequence(
        [t["seq_tensor"] for t in batch], batch_first=True, padding_value=ord(" ")
    )
    out_pos = torch.nn.utils.rnn.pad_sequence(
        [t["pos_tensor"] for t in batch], batch_first=True, padding_value=-1
    )
    out_intensity = torch.nn.utils.rnn.pad_sequence(
        [t["intensity_tensor"] for t in batch], batch_first=True, padding_value=0
    )
    out_charge = torch.stack([t["charge_tensor"] for t in batch])

    return {
        "seq_tensor": out_seqs,
        "pos_tensor": out_pos,
        "intensity_tensor": out_intensity,
        "charge_tensor": out_charge,
    }
