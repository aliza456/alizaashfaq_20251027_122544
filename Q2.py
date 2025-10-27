def entropy(y) -> float:
    """
    Compute Shannon entropy (base-2) of a label list.
    If all labels are identical (or list is empty), return 0.0.
    Result is rounded to 4 decimal places.
    """
    n = len(y)
    if n <= 1:
        return 0.0
    # count frequencies
    counts = {}
    for lab in y:
        counts[lab] = counts.get(lab, 0) + 1
    if len(counts) == 1:
        return 0.0
    # compute entropy base-2
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            # change-of-base via / ln(2) to avoid importing math.log2 in some environments
            # but we'll just use log2 where available.
            import math
            h -= p * math.log2(p)
    return round(h, 4)
