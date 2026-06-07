#!/usr/bin/env python3
"""
figure_data_emitter.py — Enhanced
AJB Diagnostic: CSV → JSON aggregation with three algorithmic enhancements.

  1A. P² (Jain–Chlamtac) online quantile estimator
  1B. Mann–Kendall monotonic trend test
  1C. Modified Z-score (MAD-based) outlier removal
"""

import math
import itertools
from typing import List, Dict, Tuple, Optional

# ════════════════════════════════════════════════════════════════════
# 1A.  P² Algorithm  —  Jain & Chlamtac (1985)
#      Online estimation of an arbitrary quantile without storing
#      the full data stream.  Memory: O(1) per quantile.
# ════════════════════════════════════════════════════════════════════

class P2QuantileEstimator:
    """
    Implements the P² algorithm for a single target quantile *p*.

    The algorithm maintains exactly 5 markers (heights q[0..4]) and
    tracks their actual positions n[0..4] alongside desired positions
    n'[0..4].  After every new observation the markers are shifted
    toward their desired positions using piecewise-parabolic (P²)
    interpolation, falling back to linear when parabolics would
    violate monotonicity.

    Reference
    ---------
    R. Jain and I. Chlamtac, "The P² Algorithm for Dynamic
    Calculation of Quantiles and Histograms Without Storing
    Observations," *Communications of the ACM*, 28(10), 1985.
    """

    def __init__(self, p: float):
        """
        Parameters
        ----------
        p : float in (0, 1)
            The quantile to track (e.g. 0.5 for the median).
        """
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0,1), got {p}")
        self.p = p

        # Desired positions (0-indexed) for the 5 markers
        self.dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

        # Will be populated after the first 5 observations
        self.q: List[float] = []       # marker heights
        self.n: List[int] = []         # actual positions (1-based)
        self.np: List[float] = []      # desired positions (1-based)
        self._init_buf: List[float] = []
        self._initialized = False
        self.count = 0

    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        """Sort the first 5 observations and seed the markers."""
        self._init_buf.sort()
        self.q = list(self._init_buf)
        self.n = [1, 2, 3, 4, 5]
        self.np = [1.0,
                   1.0 + 2.0 * self.p,
                   1.0 + 4.0 * self.p,
                   3.0 + 2.0 * self.p,
                   5.0]
        self._initialized = True
        print(f"    [P²-init] p={self.p:.2f}  seeds={self.q}"
              f"  n={self.n}  n'={[f'{v:.2f}' for v in self.np]}")

    # ------------------------------------------------------------------
    @staticmethod
    def _parabolic(d: int, qi: float, qim1: float, qip1: float,
                   ni: int, nim1: int, nip1: int) -> float:
        """Piecewise-parabolic (P²) interpolation formula."""
        a = d / (nip1 - nim1)
        b = ((ni - nim1 + d) * (qip1 - qi) / (nip1 - ni)
             + (nip1 - ni - d) * (qi - qim1) / (ni - nim1))
        return qi + a * b

    # ------------------------------------------------------------------
    def observe(self, x: float) -> None:
        """Feed one observation into the estimator."""
        self.count += 1

        # --- Phase 1: collect the first 5 points ---
        if not self._initialized:
            self._init_buf.append(x)
            if len(self._init_buf) == 5:
                self._initialize()
            return

        # --- Phase 2: steady-state update ---
        # (a) find the cell k such that q[k-1] <= x < q[k]
        if x < self.q[0]:
            self.q[0] = x
            k = 1
        elif x < self.q[1]:
            k = 1
        elif x < self.q[2]:
            k = 2
        elif x < self.q[3]:
            k = 3
        elif x <= self.q[4]:
            k = 4
        else:
            self.q[4] = x
            k = 4

        # (b) increment positions of markers k+1 … 4
        for i in range(k, 5):
            self.n[i] += 1

        # (c) update desired positions
        for i in range(5):
            self.np[i] += self.dn[i]

        # (d) adjust marker heights for markers 1,2,3
        for i in (1, 2, 3):
            d_i = self.np[i] - self.n[i]
            if ((d_i >= 1.0 and self.n[i + 1] - self.n[i] > 1) or
                    (d_i <= -1.0 and self.n[i - 1] - self.n[i] < -1)):
                d = 1 if d_i > 0 else -1
                q_new = self._parabolic(
                    d, self.q[i], self.q[i - 1], self.q[i + 1],
                    self.n[i], self.n[i - 1], self.n[i + 1])
                # Monotonicity guard → fall back to linear
                if self.q[i - 1] < q_new < self.q[i + 1]:
                    self.q[i] = q_new
                else:
                    # Linear interpolation
                    idx = i + d
                    self.q[i] += (d * (self.q[idx] - self.q[i])
                                  / (self.n[idx] - self.n[i]))
                self.n[i] += d

    # ------------------------------------------------------------------
    @property
    def result(self) -> Optional[float]:
        """Return the current quantile estimate (marker 2)."""
        if not self._initialized:
            # Fewer than 5 observations: exact quantile of buffer
            if not self._init_buf:
                return None
            s = sorted(self._init_buf)
            idx = self.p * (len(s) - 1)
            lo = int(math.floor(idx))
            hi = min(lo + 1, len(s) - 1)
            frac = idx - lo
            return s[lo] * (1 - frac) + s[hi] * frac
        return self.q[2]


class P2TripleTracker:
    """
    Convenience wrapper: tracks p25, p50, p75 simultaneously
    for a single (method, x-point) bucket.
    """

    def __init__(self):
        self.estimators = {
            'p25': P2QuantileEstimator(0.25),
            'p50': P2QuantileEstimator(0.50),
            'p75': P2QuantileEstimator(0.75),
        }
        self._n = 0

    def observe(self, x: float) -> None:
        self._n += 1
        for est in self.estimators.values():
            est.observe(x)

    def snapshot(self) -> Dict[str, Optional[float]]:
        return {k: v.result for k, v in self.estimators.items()}

    @property
    def count(self) -> int:
        return self._n


def p2_quantiles_by_method(
    data: Dict[str, Dict[float, List[float]]]
) -> Dict[str, Dict[float, Dict[str, Optional[float]]]]:
    """
    Run P² quantile tracking on structured data.

    Parameters
    ----------
    data : {method_name: {x_point: [y_values, ...]}}

    Returns
    -------
    {method_name: {x_point: {'p25': ..., 'p50': ..., 'p75': ...}}}
    """
    print("=" * 72)
    print("  1A · P² Online Quantile Tracking (Jain–Chlamtac)")
    print("=" * 72)

    results: Dict[str, Dict[float, Dict[str, Optional[float]]]] = {}

    for method, xpoints in data.items():
        print(f"\n  ▸ Method: {method}")
        results[method] = {}
        for xp in sorted(xpoints.keys()):
            tracker = P2TripleTracker()
            samples = xpoints[xp]
            for val in samples:
                tracker.observe(val)
            snap = tracker.snapshot()
            results[method][xp] = snap
            print(f"    x={xp:>8.2f}  n={tracker.count:>5d}"
                  f"  p25={snap['p25']!s:>12s}"
                  f"  p50={snap['p50']!s:>12s}"
                  f"  p75={snap['p75']!s:>12s}")

    print("\n  ✓ P² quantile tracking complete.\n")
    return results


# ════════════════════════════════════════════════════════════════════
# 1B.  Mann–Kendall Trend Test
#      Non-parametric monotonic-trend detector.
# ════════════════════════════════════════════════════════════════════

def _mann_kendall(y: List[float]) -> Tuple[float, float, str]:
    """
    Compute the Mann–Kendall trend statistic for a time-ordered
    sequence *y*.

    Returns
    -------
    tau   : Kendall tau-b correlation (normalised S)
    p     : two-sided p-value (normal approximation, with
            variance correction for ties)
    label : 'trending_up' | 'trending_down' | 'no_trend'
            (at α = 0.05)

    Algorithm
    ---------
    S = Σ_{i<j} sgn(y_j − y_i)

    Under H₀ (no trend) and with tied groups of sizes t_k:
        Var(S) = [n(n−1)(2n+5) − Σ t_k(t_k−1)(2t_k+5)] / 18

    Z = (S − sign(S)) / √Var(S)   (continuity correction)
    """
    n = len(y)
    if n < 3:
        return 0.0, 1.0, "no_trend"

    # Compute S
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
            # ties contribute 0

    print(f"      S = {s}  (n={n})")

    # Tied-group sizes
    from collections import Counter
    counts = Counter(y)
    tie_groups = [c for c in counts.values() if c > 1]

    # Variance
    var_s = n * (n - 1) * (2 * n + 5)
    for t in tie_groups:
        var_s -= t * (t - 1) * (2 * t + 5)
    var_s /= 18.0

    print(f"      Var(S) = {var_s:.4f}  tie_groups={tie_groups}")

    if var_s == 0:
        return 0.0, 1.0, "no_trend"

    # Continuity-corrected Z
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value (standard normal CDF via error function)
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # Kendall tau-b
    n_pairs = n * (n - 1) / 2.0
    # concordant-discordant pairs adjusted for ties
    t_adjust = sum(t * (t - 1) / 2.0 for t in counts.values())
    denom = math.sqrt((n_pairs - t_adjust) * n_pairs) if (n_pairs - t_adjust) > 0 else 1.0
    tau = s / denom

    if p_value < 0.05:
        label = "trending_up" if s > 0 else "trending_down"
    else:
        label = "no_trend"

    print(f"      Z = {z:.4f}  p = {p_value:.6f}  τ = {tau:.4f}  → {label}")

    return tau, p_value, label


def mann_kendall_by_method(
    data: Dict[str, List[float]]
) -> Dict[str, Dict[str, object]]:
    """
    Run Mann–Kendall on every method's Y series.

    Parameters
    ----------
    data : {method_name: [y_values ordered by x]}

    Returns
    -------
    {method_name: {'tau': float, 'p': float, 'label': str}}
    """
    print("=" * 72)
    print("  1B · Mann–Kendall Monotonic Trend Test")
    print("=" * 72)

    results = {}
    for method, y_series in data.items():
        print(f"\n  ▸ Method: {method}  (series length {len(y_series)})")
        print(f"    first 10 values: {y_series[:10]}")
        tau, p, label = _mann_kendall(y_series)
        results[method] = {'tau': tau, 'p_value': p, 'trend_label': label}

    print("\n  ✓ Mann–Kendall analysis complete.\n")
    return results


# ════════════════════════════════════════════════════════════════════
# 1C.  Modified Z-score Outlier Removal  (Iglewicz & Hoaglin, 1993)
#      Uses Median Absolute Deviation (MAD) instead of σ.
# ════════════════════════════════════════════════════════════════════

def _median(xs: List[float]) -> float:
    """Compute the median of a sorted or unsorted list."""
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def modified_zscore_filter(
    samples: List[float],
    threshold: float = 3.5
) -> Tuple[List[float], List[float]]:
    """
    Remove outliers from *samples* using the Modified Z-score.

    Modified Z_i = 0.6745 · (x_i − median) / MAD

    where MAD = median(|x_i − median|) and 0.6745 ≈ Φ⁻¹(0.75),
    making the score comparable to a standard Z-score for normal
    data.

    Parameters
    ----------
    samples   : raw observations
    threshold : cutoff (default 3.5, per Iglewicz & Hoaglin)

    Returns
    -------
    (kept, removed) : two lists
    """
    if len(samples) < 3:
        return list(samples), []

    med = _median(samples)
    abs_devs = [abs(x - med) for x in samples]
    mad = _median(abs_devs)

    print(f"      median={med:.6f}  MAD={mad:.6f}  n={len(samples)}")

    if mad == 0.0:
        # All values identical (or nearly); no outliers
        print("      MAD=0 → no outliers detectable")
        return list(samples), []

    kept, removed = [], []
    for x in samples:
        mz = 0.6745 * (x - med) / mad
        if abs(mz) > threshold:
            removed.append(x)
        else:
            kept.append(x)

    if removed:
        print(f"      removed {len(removed)} outlier(s): {removed[:10]}"
              f"{'…' if len(removed) > 10 else ''}")
    else:
        print(f"      no outliers (all |M_i| ≤ {threshold})")

    return kept, removed


def filter_outliers_by_method(
    data: Dict[str, Dict[float, List[float]]],
    threshold: float = 3.5
) -> Dict[str, Dict[float, List[float]]]:
    """
    Apply Modified Z-score outlier removal to every
    (method, x-point) bucket.

    Parameters
    ----------
    data : {method: {x_point: [raw_samples]}}

    Returns
    -------
    Same structure with outliers removed.
    """
    print("=" * 72)
    print("  1C · Modified Z-score Outlier Removal (MAD, threshold="
          f"{threshold})")
    print("=" * 72)

    total_removed = 0
    total_samples = 0
    cleaned: Dict[str, Dict[float, List[float]]] = {}

    for method, xpoints in data.items():
        print(f"\n  ▸ Method: {method}")
        cleaned[method] = {}
        for xp in sorted(xpoints.keys()):
            raw = xpoints[xp]
            total_samples += len(raw)
            print(f"    x={xp:>8.2f}  n_raw={len(raw):>5d}", end="  →  ")
            kept, removed = modified_zscore_filter(raw, threshold)
            total_removed += len(removed)
            cleaned[method][xp] = kept

    pct = (100.0 * total_removed / total_samples) if total_samples else 0.0
    print(f"\n  Summary: removed {total_removed}/{total_samples}"
          f" samples ({pct:.2f}%)")
    print("  ✓ Outlier filtering complete.\n")
    return cleaned


# ════════════════════════════════════════════════════════════════════
#  Demo / smoke test
# ════════════════════════════════════════════════════════════════════

def _demo():
    import random
    random.seed(42)

    # Synthetic data: 3 methods, 5 x-points, ~50 samples each
    methods = ['hash_join', 'sort_merge', 'nested_loop']
    x_points = [10.0, 20.0, 50.0, 100.0, 200.0]

    raw_data: Dict[str, Dict[float, List[float]]] = {}
    trend_data: Dict[str, List[float]] = {}

    for m_idx, method in enumerate(methods):
        raw_data[method] = {}
        y_means = []
        for xp in x_points:
            base = (m_idx + 1) * xp * 0.5
            samples = [base + random.gauss(0, base * 0.1) for _ in range(50)]
            # Inject a couple of outliers
            samples.append(base * 5.0)
            samples.append(-base * 2.0)
            raw_data[method][xp] = samples
            y_means.append(sum(samples) / len(samples))
        trend_data[method] = y_means

    # 1C — outlier removal first
    cleaned = filter_outliers_by_method(raw_data, threshold=3.5)

    # 1A — P² quantiles on cleaned data
    quantiles = p2_quantiles_by_method(cleaned)

    # 1B — Mann–Kendall on mean-aggregated Y series
    trends = mann_kendall_by_method(trend_data)

    print("═" * 72)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 72)
    for method in methods:
        t = trends[method]
        print(f"\n  {method}:  τ={t['tau']:.4f}  p={t['p_value']:.6f}"
              f"  → {t['trend_label']}")
        for xp in x_points:
            q = quantiles[method][xp]
            print(f"    x={xp:>6.1f}  p25={q['p25']!s:>12s}"
                  f"  p50={q['p50']!s:>12s}"
                  f"  p75={q['p75']!s:>12s}")


if __name__ == "__main__":
    _demo()
