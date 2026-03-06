def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    if not tokens:
        return []
    step = chunk_size - overlap
    n = len(tokens)
    return [tokens[i: i + chunk_size] for i in range(0, max(0, n - chunk_size) + 1, step)]
