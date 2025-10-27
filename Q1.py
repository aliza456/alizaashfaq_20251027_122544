def kmeans_assign(points, centroids) -> list[int]:
    """
    Assign each point to the nearest centroid by Euclidean distance.
    Break ties by choosing the smaller centroid index.
    Returns list of centroid indices (0-based).
    """
    import math

    if not centroids:
        return []

    def sqdist(a, b):
        # squared Euclidean distance
        return sum((x - y) ** 2 for x, y in zip(a, b))

    assignments = []
    for p in points:
        best_idx = 0
        best_d = sqdist(p, centroids[0])
        for j in range(1, len(centroids)):
            d = sqdist(p, centroids[j])
            if d < best_d or (d == best_d and j < best_idx):
                best_d = d
                best_idx = j
        assignments.append(best_idx)
    return assignments
