"""Independent exact-real mathematical reference for ARCHFUSE089 channel delta."""


def retained(m, alpha):
    n = len(alpha)
    if n == 0 or not m or any(len(row) != n for row in m):
        raise ValueError("value-by-key rectangular memory")
    return [[v * alpha[j] for j, v in enumerate(row)] for row in m]


def channel_delta(m, key, value, alpha, beta):
    if not 0 <= beta <= 1 or len(key) != len(alpha) or len(value) != len(m):
        raise ValueError("shapes or write strength")
    if any(not 0 <= a <= 1 for a in alpha):
        raise ValueError("retention range")
    aged = retained(m, alpha)
    error = [value[i] - sum(aged[i][j] * key[j] for j in range(len(key)))
             for i in range(len(value))]
    return [[aged[i][j] + beta * error[i] * key[j] for j in range(len(key))]
            for i in range(len(value))]
