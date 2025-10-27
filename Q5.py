def top_k_cosine(query, docs, k) -> list[int]:
    """
    Return indices of top-k documents by cosine similarity to a query vector.
    Cosine similarity = (q·d)/(||q||·||d||).
    Break ties by smaller index. Treat zero-norm vectors as similarity 0.
    """
    import math

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    def norm(v):
        return math.sqrt(sum(x*x for x in v))

    if k <= 0 or not docs:
        return []

    qn = norm(query)
    sims = []
    for idx, d in enumerate(docs):
        dn = norm(d)
        if qn == 0 or dn == 0:
            s = 0.0
        else:
            s = dot(query, d) / (qn * dn)
        sims.append((-s, idx))  # negative for descending; idx for tie-break
    sims.sort()
    k = min(k, len(docs))
    return [idx for _, idx in sims[:k]]
