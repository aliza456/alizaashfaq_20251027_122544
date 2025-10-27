def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    """
    Compute a confusion matrix.
    Rows: true labels; Columns: predicted labels; both ordered as in `labels`.
    Returns a 2D list of counts.
    """
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    M = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            M[idx[yt]][idx[yp]] += 1
    return M
