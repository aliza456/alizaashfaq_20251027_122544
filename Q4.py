def knn_predict(train_X, train_y, test_X, k) -> list[int]:
    """
    k-Nearest Neighbors prediction using Euclidean distance.
    - For each test sample, compute distance to all train samples.
    - Sort neighbors by (distance, label).
    - Take the first k, vote by majority, break label ties by choosing the smallest label.
    Return a list of predicted integer labels.
    """
    n_train = len(train_X)
    if n_train == 0:
        return [0 for _ in test_X]  # fallback if no training data

    k = min(max(1, k), n_train)

    def sqdist(a, b):
        s = 0.0
        for xa, xb in zip(a, b):
            d = xa - xb
            s += d*d
        return s

    preds = []
    for x in test_X:
        dlist = []
        for xi, yi in zip(train_X, train_y):
            d2 = sqdist(x, xi)
            dlist.append((d2, yi))
        dlist.sort(key=lambda t: (t[0], t[1]))
        k_nearest = dlist[:k]
        counts = {}
        maxc = 0
        for _, lab in k_nearest:
            counts[lab] = counts.get(lab, 0) + 1
            if counts[lab] > maxc:
                maxc = counts[lab]
        best = min(lab for lab, cnt in counts.items() if cnt == maxc)
        preds.append(int(best))
    return preds
