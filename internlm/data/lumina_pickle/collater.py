import torch

def extend_data_to_packed_length(data: list, max_length, padding):
    """Helper function to extend data to the specified length with padding."""
    if len(data) > max_length:
        return data[:max_length]
    if len(data) == max_length:
        return data
    # len(data) < max_length
    return data + [padding] * (max_length - len(data))

def lumina_collate_fn(batch):
    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".
    """
    # Initialize lists to store the data from each sample
    tokens, labels, type_ids, indexes = [], [], [], []
    cumulative_seqlens = [0]

    # Calculate the maximum sequence length
    max_length = max(len(sample["tokens"]) for sample in batch)
    max_length = max(max_length, batch[0]["seq_len"])

    # Accumulate all samples into respective lists
    for sample in batch:
        tokens.extend(extend_data_to_packed_length(sample["tokens"], max_length, 0.0))
        labels.extend(extend_data_to_packed_length(sample["labels"], max_length, -100.0))
        type_ids.extend(extend_data_to_packed_length(sample["type_ids"], max_length, 0.0))
        indexes.extend(list(range(max_length)))
        cumulative_seqlens.append(cumulative_seqlens[-1] + max_length)

    # Convert lists to tensors
    xs = torch.tensor(tokens, dtype=torch.long)
    ys = torch.tensor(labels, dtype=torch.long)
    ts = torch.tensor(type_ids, dtype=torch.long)
    indexes = torch.tensor(indexes, dtype=torch.long)
    cu_seqlens = torch.tensor(cumulative_seqlens[:-1], dtype=torch.int)

    # Create the output dictionary
    input_data = {
        "input_ids": xs,
        "cu_seqlens": cu_seqlens,
        "indexes": indexes,
        "type_ids": ts
    }

    return input_data, ys