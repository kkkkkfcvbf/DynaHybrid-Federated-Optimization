"""Small metric helpers used by the training loop."""

def weighted_mean(values, weights):
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / max(total, 1)
