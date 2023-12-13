def concatenate_text_fields(batch, text_cols):
    """
    Concatenates the text fields in a dataset.

    Args:
        batch (Dataset): A dictionary containing the batch data.
        text_cols (list): A list of column names representing the text fields.

    Returns:
        dict: The updated dataset with the "text" field containing the concatenated text.

    """
    text = []
    for idx in range(len(list(batch.values())[0])):
        concatenated_text = " ".join([str(batch[col][idx]) for col in text_cols if batch[col][idx] is not None])
        text.append(concatenated_text)
    batch["text"] = text
    return batch