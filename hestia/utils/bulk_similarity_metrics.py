import numpy as np

from numba import njit


@njit
def bulk_tanimoto_continuous(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    n, d = bulk.shape[0], bulk.shape[1]
    result = np.empty(n)

    norm_u = 0.0
    for j in range(d):
        norm_u += u[j] * u[j]

    for i in range(n):
        dot = 0.0
        norm_b = 0.0

        for j in range(d):
            dot += bulk[i, j] * u[j]
            norm_b += bulk[i, j] * bulk[i, j]

        denom = norm_u + norm_b - dot
        result[i] = dot / denom

    return result


# @njit
def bulk_jensen_shannon(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    n, d = bulk.shape
    distances = np.empty(n)

    # normalize u
    u_sum = np.sum(u)
    u_norm = u / u_sum

    for i in range(n):
        row = bulk[i]
        row_sum = np.sum(row)
        row_norm = row / row_sum

        js = 0.0
        for j in range(d):
            m = 0.5 * (u_norm[j] + row_norm[j])

            if u_norm[j] > 0:
                js += 0.5 * u_norm[j] * np.log(u_norm[j] / m)
            if row_norm[j] > 0:
                js += 0.5 * row_norm[j] * np.log(row_norm[j] / m)

        distances[i] = np.sqrt(js)

    return distances


@njit
def bulk_mahalanobis(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    n, d = bulk.shape

    # --- compute mean ---
    mean = np.zeros(d)
    for i in range(n):
        for j in range(d):
            mean[j] += bulk[i, j]
    for j in range(d):
        mean[j] /= n

    # --- compute covariance matrix ---
    cov = np.zeros((d, d))
    for i in range(n):
        for j in range(d):
            for k in range(d):
                cov[j, k] += (bulk[i, j] - mean[j]) * (bulk[i, k] - mean[k])
    for j in range(d):
        for k in range(d):
            cov[j, k] /= (n - 1)

    # --- invert covariance (do this once) ---
    inv_cov = np.linalg.inv(cov)

    # --- compute distances ---
    distances = np.empty(n)
    for i in range(n):
        # temp = inv_cov @ diff
        temp = np.zeros(d)
        for j in range(d):
            for k in range(d):
                temp[j] += inv_cov[j, k] * (bulk[i, k] - u[k])

        s = 0.0
        for j in range(d):
            diff = bulk[i, j] - u[j]
            s += temp[j] * diff

        distances[i] = np.sqrt(s)

    return distances


@njit
def bulk_np_jaccard(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    n, m = bulk.shape
    out = np.empty(n)
    for i in range(n):
        count = 0
        for j in range(m):
            if bulk[i, j] == u[j]:
                count += 1
        out[i] = count / m
    return out


def bulk_np_tanimoto(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = (a + b) - c
    return np.where(denominator == 0, 1, c / denominator)


def bulk_np_dice(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = a + b
    return np.where(denominator == 0, 1, (2 * c) / denominator)


def bulk_np_sokal(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    denominator = 2 * (a + b) - 3 * c
    return np.where(denominator == 0, 1, c / denominator)


def bulk_np_rogot_goldberg(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    a = u.sum()
    e = bulk.shape[1]
    b = bulk.sum(axis=1)
    c = np.dot(bulk, u)
    d = e + c - (a + b)
    denominator_1 = a + b
    denominator_2 = 2 * e - (a + b)
    result = np.where((a == e) | (d == e),  1, np.where(
            (denominator_1 == 0) | (denominator_2 == 0),  0, c / denominator_1 - d / denominator_2
        )
    )
    return result


def bulk_cosine_similarity(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    u_norm = u / np.linalg.norm(u)
    bulk_norm = bulk / np.linalg.norm(bulk, axis=1, keepdims=True)
    return np.dot(bulk_norm, u_norm)


def bulk_binary_manhattan_similarity(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    distances = np.abs(bulk - u).sum(axis=1)
    return distances / u.shape[0]


def bulk_euclidean(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    squared_diff = np.square(bulk - u)
    distances = np.sqrt(np.sum(squared_diff, axis=1))
    return distances


def bulk_manhattan(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    distances = np.abs(bulk - u).sum(axis=1)
    return distances


def bulk_canberra(u: np.ndarray, bulk: np.ndarray) -> np.ndarray:
    diff = np.abs(bulk - u)
    sum_abs = np.abs(bulk) + np.abs(u)
    sum_abs = np.where(sum_abs == 0, 1e-10, sum_abs)
    return np.sum(diff / sum_abs, axis=1)
