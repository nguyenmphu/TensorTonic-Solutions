def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)
    try:
        return len(a & b) / len(a | b)
    except ZeroDivisionError:
        return 0.0
