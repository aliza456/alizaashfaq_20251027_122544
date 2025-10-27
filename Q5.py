def knn_predict(train_X, train_y, test_X, k) -> list[int]:
    """
    k-Nearest Neighbors prediction using Euclidean distance.
    - For each test sample, compute distance to all train samples.
    - Sort neighbors by (distance, label).
    - Take the first k, vote by majority, break label ties by choosing the smallest label.
    Returns a list of predicted integer labels.
    """
    n_train = len(train_X)
    if n_train == 0:
        return [0 for _ in test_X]  # fallback if no training data

    # Clamp k to a sensible range
    k = min(max(1, k), n_train)

    def sqdist(a, b):
        # squared Euclidean distance (monotonic for ranking)
        s = 0.0
        for xa, xb in zip(a, b):
            d = xa - xb
            s += d * d
        return s

    preds = []
    for x in test_X:
        # Build (distance, label) list
        dl = []
        for xi, yi in zip(train_X, train_y):
            d2 = sqdist(x, xi)
            dl.append((d2, yi))
        # Sort by distance, then label for deterministic neighbor order
        dl.sort(key=lambda t: (t[0], t[1]))
        k_nearest = dl[:k]
        # Majority vote
        counts = {}
        max_count = 0
        for _, lab in k_nearest:
            counts[lab] = counts.get(lab, 0) + 1
            if counts[lab] > max_count:
                max_count = counts[lab]
        # Choose smallest label among those with max_count
        best_label = min(lab for lab, cnt in counts.items() if cnt == max_count)
        preds.append(int(best_label))
    return preds
