def perceptron_epoch(X, y, w, b, lr) -> [list[float], float]:
    """
    Single epoch of the binary perceptron.
    Labels are -1 or +1. For each sample (in order), if y_i * (w·x + b) <= 0:
        w = w + lr*y_i*x
        b = b + lr*y_i
    Returns [w, b].
    """
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    w = list(w)
    b = float(b)

    for xi, yi in zip(X, y):
        margin = yi * (dot(w, xi) + b)
        if margin <= 0:
            for j in range(len(w)):
                w[j] += lr * yi * xi[j]
            b += lr * yi
    return [w, b]
