import numpy as np

def fisher_ci(r: float, n: int, z_crit: float = 1.96) -> tuple[float, float]:
    """Approximate 95% confidence interval for a Pearson correlation (Fisher z)."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))

def permutation_pvalue(x, y, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value: how often does shuffling y give a correlation at least this strong?"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(np.corrcoef(x, y)[0, 1])
    count = sum(
        abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= observed for _ in range(n_perm)
    )
    return (count + 1) / (n_perm + 1)