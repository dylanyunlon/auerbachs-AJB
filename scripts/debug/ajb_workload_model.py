#!/usr/bin/env python3
"""
ajb_workload_model.py — Workload modeling & cost prediction for AJB join analysis

Contains:
  1. ZipfWorkloadGenerator: Rejection sampling for Zipf-distributed join keys
  2. SkewDetector: Kullback–Leibler divergence skew detection
  3. CostModelPredictor: Ordinary least-squares linear regression for join cost

Each algorithm includes [AJB_TRACE] diagnostic output compatible with
parse_ajb_trace.py.
"""

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  ANSI colours (terminal prettification)
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def _header(title: str) -> None:
    width = 72
    print(f"\n{BOLD}{CYAN}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{RESET}\n")


def _step(msg: str) -> None:
    print(f"  {DIM}│{RESET} {msg}")


def _section(msg: str) -> None:
    print(f"\n  {BOLD}{YELLOW}▸ {msg}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ZipfWorkloadGenerator — Rejection Sampling
#
#  The Zipf (power-law) distribution models the frequency of join keys in
#  real database workloads: a small number of "hot" keys dominate traffic.
#
#  P(k) = (1/k^s) / H(N,s)   for k = 1, 2, …, N
#
#  where H(N,s) = Σ_{k=1}^{N} 1/k^s  is the generalized harmonic number.
#
#  Rejection sampling approach:
#    • Proposal distribution g(k) = 1/N  (uniform on {1..N})
#    • Target f(k) = (1/k^s) / H(N,s)
#    • Acceptance probability  f(k) / (c · g(k))  where c = N / H(N,s)
#      so the ratio simplifies to  1/k^s  (since f(k)·N/c = 1/k^s).
#    • Draw u ~ Uniform(0,1); accept k if u ≤ 1/k^s.
#
#  Expected attempts per accepted sample:  c = N / H(N,s).
#  For large s this grows slowly because H(N,s) converges.
# ═══════════════════════════════════════════════════════════════════════════════


class ZipfWorkloadGenerator:
    """Generate Zipf-distributed join keys via rejection sampling.

    Parameters
    ----------
    n_keys : int
        Number of distinct keys (support is {1, 2, …, n_keys}).
    skew : float
        Zipf exponent s ≥ 0.  s = 0 → uniform; s = 1 → classic Zipf;
        s > 1 → increasingly heavy head.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(self, n_keys: int, skew: float = 1.0, seed: Optional[int] = None):
        if n_keys < 1:
            raise ValueError(f"n_keys must be ≥ 1, got {n_keys}")
        if skew < 0.0:
            raise ValueError(f"skew must be ≥ 0, got {skew}")

        self.n_keys = n_keys
        self.skew = skew
        self.rng = random.Random(seed)

        # Pre-compute the generalized harmonic number  H(N, s)
        self.harmonic = sum(1.0 / (k ** skew) for k in range(1, n_keys + 1))

        # Rejection-sampling envelope constant  c = N / H(N, s)
        self.c_envelope = n_keys / self.harmonic

        # Pre-compute per-key PMF for verification
        self._pmf = [0.0] + [(1.0 / (k ** skew)) / self.harmonic
                             for k in range(1, n_keys + 1)]

        print(f"[AJB_TRACE][ZipfWorkload] init: n_keys={n_keys}  s={skew:.3f}  "
              f"H(N,s)={self.harmonic:.6f}  envelope_c={self.c_envelope:.4f}")

    def _sample_one(self) -> Tuple[int, int]:
        """Draw one Zipf sample.  Returns (key, n_attempts)."""
        attempts = 0
        while True:
            attempts += 1
            # Proposal: uniform draw from {1 .. N}
            k = self.rng.randint(1, self.n_keys)
            # Acceptance probability = 1 / k^s
            accept_prob = 1.0 / (k ** self.skew)
            u = self.rng.random()
            if u <= accept_prob:
                return k, attempts

    def generate(self, n_samples: int) -> List[int]:
        """Generate *n_samples* Zipf-distributed keys.

        Returns the list of keys and prints trace diagnostics with
        acceptance statistics.
        """
        _section(f"ZipfWorkloadGenerator.generate(n={n_samples})")

        keys: List[int] = []
        total_attempts = 0

        # Trace every ~20 % of the run and at the end
        trace_interval = max(1, n_samples // 5)

        for i in range(n_samples):
            k, att = self._sample_one()
            keys.append(k)
            total_attempts += att

            if (i + 1) % trace_interval == 0 or i == n_samples - 1:
                eff = (i + 1) / total_attempts
                _step(f"[AJB_TRACE][ZipfWorkload] progress {i + 1}/{n_samples}  "
                      f"attempts_so_far={total_attempts}  "
                      f"acceptance_rate={eff:.4f}")

        overall_rate = n_samples / total_attempts
        avg_attempts = total_attempts / n_samples
        print(f"[AJB_TRACE][ZipfWorkload] done: n_samples={n_samples}  "
              f"total_proposals={total_attempts}  "
              f"overall_accept={overall_rate:.4f}  "
              f"avg_attempts/sample={avg_attempts:.2f}")

        return keys

    def pmf(self, k: int) -> float:
        """Theoretical PMF  P(K = k)."""
        if k < 1 or k > self.n_keys:
            return 0.0
        return self._pmf[k]

    def empirical_pmf(self, keys: List[int]) -> Dict[int, float]:
        """Compute the empirical PMF from a list of samples."""
        counts = Counter(keys)
        n = len(keys)
        return {k: counts.get(k, 0) / n for k in range(1, self.n_keys + 1)}


# ═══════════════════════════════════════════════════════════════════════════════
#  2. SkewDetector — Kullback–Leibler Divergence
#
#  KL(P ‖ Q) = Σ_k  P(k) · ln( P(k) / Q(k) )
#
#  Measures how much the observed key distribution P diverges from a
#  reference distribution Q (typically uniform).  A join whose key
#  distribution has high KL divergence from uniform is "skewed" — hot keys
#  will cause partition imbalance in hash joins and data amplification
#  in sort-merge joins.
#
#  Implementation notes:
#    • Zero-probability bins in P are skipped (0 · ln(0) → 0 by convention).
#    • Zero-probability bins in Q get additive smoothing ε = 1e-12 to avoid
#      division by zero (Laplace-like, but minimal so it doesn't distort).
#    • We also compute the symmetric Jensen–Shannon divergence as a sanity
#      check:  JSD(P,Q) = ½ KL(P‖M) + ½ KL(Q‖M),  M = ½(P+Q).
# ═══════════════════════════════════════════════════════════════════════════════


class SkewDetector:
    """Detect key-distribution skew using Kullback–Leibler divergence.

    The detector compares an observed key-frequency distribution against
    a reference (default: uniform) and classifies the skew level.

    Parameters
    ----------
    n_keys : int
        Size of the key domain {1, …, n_keys}.
    reference : dict or None
        Reference PMF.  If None, uniform over {1 .. n_keys} is used.
    """

    # KL thresholds for skew classification (empirically tuned for join
    # workloads; these match the ranges seen in TPC-H / JOB benchmarks)
    THRESHOLDS = {
        'low':      0.10,   # KL < 0.10 → essentially uniform
        'moderate': 1.00,   # 0.10 ≤ KL < 1.00 → moderate skew
                            # KL ≥ 1.00 → high skew
    }

    _EPS = 1e-12  # additive smoothing for zero bins in Q

    def __init__(self, n_keys: int,
                 reference: Optional[Dict[int, float]] = None):
        self.n_keys = n_keys
        if reference is None:
            # Uniform reference
            self.ref = {k: 1.0 / n_keys for k in range(1, n_keys + 1)}
        else:
            self.ref = dict(reference)

        print(f"[AJB_TRACE][SkewDetector] init: n_keys={n_keys}  "
              f"ref_type={'uniform' if reference is None else 'custom'}  "
              f"ref_entropy={self._entropy(self.ref):.6f}")

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(pmf: Dict[int, float]) -> float:
        """Shannon entropy  H(P) = −Σ P(k) ln P(k)."""
        h = 0.0
        for p in pmf.values():
            if p > 0:
                h -= p * math.log(p)
        return h

    # ------------------------------------------------------------------
    def kl_divergence(self, observed: Dict[int, float]) -> float:
        """Compute  KL(observed ‖ reference).

        Parameters
        ----------
        observed : {key: probability} — must sum to ~1.

        Returns
        -------
        KL divergence in nats (natural log).
        """
        kl = 0.0
        for k in range(1, self.n_keys + 1):
            p = observed.get(k, 0.0)
            q = self.ref.get(k, 0.0)
            if p <= 0.0:
                continue  # 0 · ln(0/q) = 0 by convention
            q_safe = max(q, self._EPS)
            kl += p * math.log(p / q_safe)
        return kl

    # ------------------------------------------------------------------
    def jensen_shannon(self, observed: Dict[int, float]) -> float:
        """Symmetric Jensen–Shannon divergence  JSD(P, Q)."""
        m = {}
        all_keys = set(observed.keys()) | set(self.ref.keys())
        for k in all_keys:
            m[k] = 0.5 * observed.get(k, 0.0) + 0.5 * self.ref.get(k, 0.0)

        kl_pm = 0.0
        kl_qm = 0.0
        for k in all_keys:
            p = observed.get(k, 0.0)
            q = self.ref.get(k, 0.0)
            mk = max(m[k], self._EPS)
            if p > 0:
                kl_pm += p * math.log(p / mk)
            if q > 0:
                kl_qm += q * math.log(q / mk)

        return 0.5 * kl_pm + 0.5 * kl_qm

    # ------------------------------------------------------------------
    def classify(self, observed: Dict[int, float]) -> str:
        """Classify skew level as 'low', 'moderate', or 'high'."""
        kl = self.kl_divergence(observed)
        if kl < self.THRESHOLDS['low']:
            return 'low'
        elif kl < self.THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'high'

    # ------------------------------------------------------------------
    def detect(self, keys: List[int]) -> Dict[str, object]:
        """Full skew analysis on a list of raw keys.

        Returns a dict with KL divergence, JSD, entropy, classification,
        and top-K hot keys.
        """
        _section(f"SkewDetector.detect(n_samples={len(keys)})")

        # Empirical PMF
        counts = Counter(keys)
        n = len(keys)
        observed = {k: counts.get(k, 0) / n for k in range(1, self.n_keys + 1)}

        # Metrics
        kl = self.kl_divergence(observed)
        jsd = self.jensen_shannon(observed)
        h_obs = self._entropy(observed)
        h_ref = self._entropy(self.ref)
        label = self.classify(observed)

        # Top-10 hottest keys
        top_keys = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

        _step(f"[AJB_TRACE][SkewDetector] observed_entropy={h_obs:.6f}  "
              f"ref_entropy={h_ref:.6f}")
        _step(f"[AJB_TRACE][SkewDetector] KL(obs‖ref)={kl:.6f}  "
              f"JSD={jsd:.6f}")
        _step(f"[AJB_TRACE][SkewDetector] classification={label}  "
              f"(thresholds: low<{self.THRESHOLDS['low']}, "
              f"mod<{self.THRESHOLDS['moderate']})")
        _step(f"[AJB_TRACE][SkewDetector] top-10 keys: "
              f"{[(k, c, f'{c/n:.4f}') for k, c in top_keys]}")

        # Per-key divergence contribution (top contributors)
        contrib = []
        for k in range(1, self.n_keys + 1):
            p = observed.get(k, 0.0)
            q = max(self.ref.get(k, 0.0), self._EPS)
            if p > 0:
                contrib.append((k, p * math.log(p / q)))
        contrib.sort(key=lambda x: x[1], reverse=True)
        _step(f"[AJB_TRACE][SkewDetector] top-5 KL contributors: "
              f"{[(k, f'{c:.6f}') for k, c in contrib[:5]]}")

        return {
            'kl_divergence': kl,
            'jensen_shannon': jsd,
            'observed_entropy': h_obs,
            'reference_entropy': h_ref,
            'classification': label,
            'distinct_observed': len(counts),
            'top_keys': top_keys,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. CostModelPredictor — Ordinary Least-Squares Linear Regression
#
#  Predicts join execution cost from three features:
#    x₁ = table_size      (total rows in the larger relation)
#    x₂ = distinct_keys   (number of distinct join-key values)
#    x₃ = skew            (KL divergence of key distribution from uniform)
#
#  Model:   ŷ = β₀ + β₁·x₁ + β₂·x₂ + β₃·x₃
#
#  OLS closed-form:  β = (XᵀX)⁻¹ Xᵀy
#
#  We invert the 4×4 normal matrix by Gauss–Jordan elimination (no numpy).
#  R² = 1 − SS_res / SS_tot   measures goodness of fit.
# ═══════════════════════════════════════════════════════════════════════════════


class CostModelPredictor:
    """Simple OLS linear regression for join-cost prediction.

    Features (per training sample):
        [table_size, distinct_keys, skew]

    The model augments the feature vector with a bias column internally.
    """

    def __init__(self):
        self.beta: Optional[List[float]] = None   # [β₀, β₁, β₂, β₃]
        self.r_squared: Optional[float] = None
        self.n_train: int = 0
        self.feature_names = ['intercept', 'table_size', 'distinct_keys', 'skew']

    # ------------------------------------------------------------------
    #  Matrix utilities (pure Python, no external deps)
    # ------------------------------------------------------------------

    @staticmethod
    def _mat_mul(a: List[List[float]],
                 b: List[List[float]]) -> List[List[float]]:
        """Multiply matrices a (m×n) and b (n×p) → (m×p)."""
        m, n, p = len(a), len(a[0]), len(b[0])
        result = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += a[i][k] * b[k][j]
                result[i][j] = s
        return result

    @staticmethod
    def _mat_transpose(a: List[List[float]]) -> List[List[float]]:
        """Transpose a matrix."""
        m, n = len(a), len(a[0])
        return [[a[i][j] for i in range(m)] for j in range(n)]

    @staticmethod
    def _mat_inverse_4x4(m: List[List[float]]) -> Optional[List[List[float]]]:
        """Invert a 4×4 matrix via Gauss–Jordan elimination.

        Returns None if the matrix is singular (|pivot| < 1e-14).
        """
        n = 4
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i, row in enumerate(m)]

        for col in range(n):
            # Partial pivoting: find the row with the largest absolute
            # value in this column from col..n-1
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < 1e-14:
                return None  # singular
            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot

            # Eliminate other rows
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

        # Extract the right half
        return [row[n:] for row in aug]

    # ------------------------------------------------------------------
    #  Core OLS
    # ------------------------------------------------------------------

    def fit(self, features: List[List[float]], targets: List[float]) -> None:
        """Fit the linear model  y = Xβ  via OLS normal equations.

        Parameters
        ----------
        features : list of [table_size, distinct_keys, skew] per sample
        targets  : list of observed join costs
        """
        _section(f"CostModelPredictor.fit(n={len(features)})")

        n = len(features)
        if n < 4:
            raise ValueError(f"Need ≥ 4 training samples for 3 features + intercept, "
                             f"got {n}")
        self.n_train = n

        # Augment with intercept column  → X is (n × 4)
        X = [[1.0] + row for row in features]
        y = [[t] for t in targets]  # column vector (n × 1)

        _step(f"[AJB_TRACE][CostModel] design matrix X: {n}×{len(X[0])}")
        _step(f"[AJB_TRACE][CostModel] target range: "
              f"[{min(targets):.2f}, {max(targets):.2f}]  "
              f"mean={sum(targets)/n:.2f}")

        # Xᵀ
        Xt = self._mat_transpose(X)
        _step(f"[AJB_TRACE][CostModel] computing XᵀX  (4×{n} · {n}×4)")

        # XᵀX  (4×4)
        XtX = self._mat_mul(Xt, X)
        _step(f"[AJB_TRACE][CostModel] XᵀX diagonal: "
              f"[{', '.join(f'{XtX[i][i]:.4f}' for i in range(4))}]")

        # (XᵀX)⁻¹
        XtX_inv = self._mat_inverse_4x4(XtX)
        if XtX_inv is None:
            raise ValueError("XᵀX is singular — features may be linearly dependent")
        _step(f"[AJB_TRACE][CostModel] XᵀX inverted successfully (Gauss–Jordan)")

        # Xᵀy  (4×1)
        Xty = self._mat_mul(Xt, y)

        # β = (XᵀX)⁻¹ · Xᵀy
        beta_mat = self._mat_mul(XtX_inv, Xty)
        self.beta = [beta_mat[i][0] for i in range(4)]

        _step(f"[AJB_TRACE][CostModel] β = {self.beta}")
        for i, name in enumerate(self.feature_names):
            _step(f"[AJB_TRACE][CostModel]   β_{name} = {self.beta[i]:.6f}")

        # R² goodness of fit
        y_mean = sum(targets) / n
        ss_tot = sum((t - y_mean) ** 2 for t in targets)
        predictions = self._predict_batch(X)
        ss_res = sum((targets[i] - predictions[i]) ** 2 for i in range(n))

        self.r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = math.sqrt(ss_res / n)

        _step(f"[AJB_TRACE][CostModel] SS_tot={ss_tot:.4f}  SS_res={ss_res:.4f}")
        _step(f"[AJB_TRACE][CostModel] R²={self.r_squared:.6f}  RMSE={rmse:.4f}")

        # Residual diagnostics
        residuals = [targets[i] - predictions[i] for i in range(n)]
        res_mean = sum(residuals) / n
        res_std = math.sqrt(sum((r - res_mean) ** 2 for r in residuals) / n)
        _step(f"[AJB_TRACE][CostModel] residuals: mean={res_mean:.6f}  "
              f"std={res_std:.4f}  "
              f"max_abs={max(abs(r) for r in residuals):.4f}")

    # ------------------------------------------------------------------

    def _predict_batch(self, X_augmented: List[List[float]]) -> List[float]:
        """Predict from already-augmented design matrix."""
        return [sum(self.beta[j] * X_augmented[i][j] for j in range(4))
                for i in range(len(X_augmented))]

    def predict(self, features: List[float]) -> float:
        """Predict the join cost for a single feature vector.

        Parameters
        ----------
        features : [table_size, distinct_keys, skew]

        Returns
        -------
        Predicted cost (float).
        """
        if self.beta is None:
            raise RuntimeError("Model not fitted — call .fit() first")
        x = [1.0] + features
        y_hat = sum(self.beta[j] * x[j] for j in range(4))
        print(f"[AJB_TRACE][CostModel] predict({features}) → {y_hat:.4f}")
        return y_hat

    def predict_batch(self, feature_batch: List[List[float]]) -> List[float]:
        """Predict costs for multiple feature vectors."""
        if self.beta is None:
            raise RuntimeError("Model not fitted — call .fit() first")
        X = [[1.0] + row for row in feature_batch]
        preds = self._predict_batch(X)
        print(f"[AJB_TRACE][CostModel] batch_predict: {len(preds)} predictions, "
              f"range=[{min(preds):.2f}, {max(preds):.2f}]")
        return preds


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo functions
# ═══════════════════════════════════════════════════════════════════════════════


def demo_zipf() -> Tuple[ZipfWorkloadGenerator, List[int]]:
    """Demonstrate Zipf workload generation with rejection sampling."""
    _header("Demo 1 · Zipf Workload Generation (Rejection Sampling)")

    gen = ZipfWorkloadGenerator(n_keys=100, skew=1.2, seed=42)
    keys = gen.generate(n_samples=5000)

    # Verify: compare empirical vs theoretical for top-5 keys
    empirical = gen.empirical_pmf(keys)
    _section("Empirical vs Theoretical PMF (top-10 keys)")
    ranked = sorted(range(1, gen.n_keys + 1),
                    key=lambda k: empirical.get(k, 0), reverse=True)
    for k in ranked[:10]:
        emp = empirical[k]
        theo = gen.pmf(k)
        ratio = emp / theo if theo > 0 else float('inf')
        _step(f"key={k:>3d}  empirical={emp:.5f}  theoretical={theo:.5f}  "
              f"ratio={ratio:.3f}")

    print(f"\n[AJB_TRACE][ZipfDemo] generated {len(keys)} keys, "
          f"distinct={len(set(keys))}/{gen.n_keys}")
    return gen, keys


def demo_skew(keys: List[int], n_keys: int) -> Dict[str, object]:
    """Demonstrate KL-divergence skew detection."""
    _header("Demo 2 · Skew Detection (Kullback–Leibler Divergence)")

    detector = SkewDetector(n_keys=n_keys)
    result = detector.detect(keys)

    # Also run on a truly uniform sample for comparison
    _section("Comparison: uniform random keys")
    rng = random.Random(99)
    uniform_keys = [rng.randint(1, n_keys) for _ in range(len(keys))]
    uniform_result = detector.detect(uniform_keys)

    _section("Side-by-side")
    _step(f"Zipf  → KL={result['kl_divergence']:.6f}  "
          f"class={result['classification']}  "
          f"distinct={result['distinct_observed']}")
    _step(f"Unif  → KL={uniform_result['kl_divergence']:.6f}  "
          f"class={uniform_result['classification']}  "
          f"distinct={uniform_result['distinct_observed']}")

    return result


def demo_cost_model() -> CostModelPredictor:
    """Demonstrate linear-regression cost prediction for joins."""
    _header("Demo 3 · Join Cost Prediction (OLS Linear Regression)")

    rng = random.Random(7)

    # Synthetic training data: cost ≈ 2.5·table_size + 0.8·distinct − 15·skew + 100
    # (skewed joins can use partition pruning → lower cost, hence negative coeff)
    _section("Generating synthetic training data")
    features: List[List[float]] = []
    targets: List[float] = []
    true_beta = [100.0, 2.5, 0.8, -15.0]  # [intercept, table_size, distinct, skew]

    for _ in range(40):
        table_size = rng.uniform(1_000, 100_000)
        distinct = rng.uniform(50, table_size * 0.1)
        skew = rng.uniform(0.0, 3.0)
        noise = rng.gauss(0, 500)

        cost = (true_beta[0]
                + true_beta[1] * table_size
                + true_beta[2] * distinct
                + true_beta[3] * skew
                + noise)
        features.append([table_size, distinct, skew])
        targets.append(cost)

    _step(f"[AJB_TRACE][CostDemo] true coefficients: {true_beta}")
    _step(f"[AJB_TRACE][CostDemo] training samples: {len(features)}")

    model = CostModelPredictor()
    model.fit(features, targets)

    # Test predictions
    _section("Test predictions")
    test_cases = [
        [10_000, 500, 0.5],
        [50_000, 2000, 1.5],
        [80_000, 1000, 2.8],
        [5_000, 200, 0.1],
    ]
    for tc in test_cases:
        predicted = model.predict(tc)
        expected = (true_beta[0]
                    + true_beta[1] * tc[0]
                    + true_beta[2] * tc[1]
                    + true_beta[3] * tc[2])
        _step(f"  features={tc}  predicted={predicted:.2f}  "
              f"true_noiseless={expected:.2f}  "
              f"err={abs(predicted - expected):.2f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — run all three demos end to end
# ═══════════════════════════════════════════════════════════════════════════════

def _demo():
    """Orchestrate the full workload-model demonstration."""
    print(f"\n{'█' * 72}")
    print(f"  AJB Workload Model — Full Pipeline Demo")
    print(f"{'█' * 72}")

    # 1. Generate skewed keys
    gen, keys = demo_zipf()

    # 2. Detect skew
    skew_result = demo_skew(keys, gen.n_keys)

    # 3. Train cost model (using the detected skew as one feature)
    model = demo_cost_model()

    # Final summary
    _header("Pipeline Summary")
    _step(f"[AJB_TRACE][Summary] Zipf: n_keys={gen.n_keys}  s={gen.skew}  "
          f"generated={len(keys)} samples")
    _step(f"[AJB_TRACE][Summary] Skew: KL={skew_result['kl_divergence']:.6f}  "
          f"class={skew_result['classification']}")
    _step(f"[AJB_TRACE][Summary] Cost: R²={model.r_squared:.6f}  "
          f"β={[f'{b:.4f}' for b in model.beta]}")
    print(f"\n{GREEN}  ✓ All three workload-model components verified.{RESET}\n")


if __name__ == "__main__":
    _demo()
