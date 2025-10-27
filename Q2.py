def bow_transform(corpus, vocab) -> list[list[int]]:
    """
    Simple bag-of-words transformer.
    Tokenize by whitespace (lowercase assumed), count each vocab term per document.
    Return a 2D list of counts with shape [len(corpus)][len(vocab)].
    Terms not in vocab are ignored.
    """
    V = len(vocab)
    index = {term: j for j, term in enumerate(vocab)}
    out = []
    for doc in corpus:
        counts = [0] * V
        for tok in doc.split():
            j = index.get(tok)
            if j is not None:
                counts[j] += 1
        out.append(counts)
    return out
