def minmax_scale(X) -> list[list[float]]:
    """
    Min-max scale each column of X to [0,1]. For column j:
        (x - min_j) / (max_j - min_j)
    If max_j == min_j, return zeros for that column.
    Round each value to 4 decimals.
    X is a list of rows (M x N).
    """
    if not X:
        return []
    M = len(X)
    N = len(X[0]) if M > 0 else 0
    if N == 0:
        return [[] for _ in range(M)]
    # compute per-column min and max
    mins = [float('inf')] * N
    maxs = [float('-inf')] * N
    for i in range(M):
        row = X[i]
        for j in range(N):
            v = row[j]
            if v < mins[j]:
                mins[j] = v
            if v > maxs[j]:
                maxs[j] = v
    # scale
    out = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            lo, hi = mins[j], maxs[j]
            if hi == lo:
                val = 0.0
            else:
                val = (X[i][j] - lo) / (hi - lo)
            out[i][j] = round(val, 4)
    return out
