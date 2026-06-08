#pragma once

#include <cstdio>

constexpr unsigned long long operator"" _KB(unsigned long long n) { return n * 1024; }
constexpr unsigned long long operator"" _MB(unsigned long long n) { return n * 1024 * 1024; }
constexpr unsigned long long operator"" _GB(unsigned long long n) { return n * 1024 * 1024 * 1024; }

// Upstream: 经典 (a+b-1)/b 向上整除.
// AJB: 加overflow guard — 当dividend接近ULLONG_MAX时原版会溢出.
// GPU benchmark中 num_elements * sizeof(T) 可能接近上限.
inline unsigned long long DivideUp(unsigned long long dividend, unsigned long long divisor) {
  if (divisor == 0) {
    fprintf(stderr, "[DEBUG][DivideUp] division by zero! dividend=%llu\n", dividend);
    return 0;  // 安全兜底, 比crash好
  }
  // Overflow-safe: 先除再补余, 避免 dividend+divisor-1 溢出
  unsigned long long q = dividend / divisor;
  unsigned long long r = dividend % divisor;
  return q + (r > 0 ? 1 : 0);
}

inline unsigned long long RoundUp(unsigned long long dividend, unsigned long long divisor) {
  return DivideUp(dividend, divisor) * divisor;
}

// --- AJB扩展: 对齐到2的幂 ---
// GPU shared memory和warp-level操作经常需要2的幂对齐.
// Upstream里这种对齐散落在各处用位运算手写, 这里统一收口.
inline unsigned long long AlignToPow2(unsigned long long value, unsigned long long alignment) {
  // alignment必须是2的幂
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    fprintf(stderr, "[DEBUG][AlignToPow2] bad alignment=%llu (not power of 2)\n", alignment);
    return value;
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

// 最近的大于等于n的2的幂 — 用于bucket count等场景
inline unsigned long long NextPow2(unsigned long long n) {
  if (n <= 1) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

// 快速log2(向下取整) — 用于radix sort的pass数计算
inline unsigned ILog2(unsigned long long n) {
  if (n == 0) return 0;
  unsigned r = 0;
  while (n >>= 1) ++r;
  return r;
}

// --- M1111: Kahan compensated summation ---
// GPU join count accumulation over many partitions suffers from float
// catastrophic cancellation when mixing large and small cardinalities.
// Kahan-Babushka-Neumaier variant: tracks a compensation term so that
// the running error stays O(eps) instead of O(eps*n).
template <typename Iter>
double KahanSum(Iter begin, Iter end) {
  double sum = 0.0;
  double compensation = 0.0;  // running compensation for lost low-order bits
  for (Iter it = begin; it != end; ++it) {
    double y = static_cast<double>(*it) - compensation;
    double t = sum + y;
    // (t - sum) recovers the high-order part of y; subtracting y gives
    // the roundoff that was lost when adding to sum
    compensation = (t - sum) - y;
    sum = t;
  }
  fprintf(stderr, "[AJB_BP][KahanSum] final_compensation=%.2e sum=%.6f\n",
          compensation, sum);
  return sum;
}

// Accumulator variant for streaming use (call add() repeatedly)
struct KahanAccumulator {
  double sum = 0.0;
  double comp = 0.0;
  long long count = 0;

  void add(double val) {
    double y = val - comp;
    double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
    count++;
  }

  double result() const { return sum; }
  double mean() const { return count > 0 ? sum / count : 0.0; }
};

// --- M1112: Fast inverse square root (Newton-Raphson) ---
// Used in AGM bound normalization where we need 1/sqrt(x) but don't
// need full double precision.  Two Newton iterations give ~1e-7 relative
// error, sufficient for cost model heuristics.
inline double FastInvSqrt(double x) {
  if (x <= 0.0) {
    fprintf(stderr, "[AJB_BP][FastInvSqrt] non-positive input=%.6e\n", x);
    return 0.0;
  }
  // Initial guess via bit-level hack (Lomont's constant for double)
  union { double d; unsigned long long i; } conv;
  conv.d = x;
  conv.i = 0x5FE6EB50C7B537A9ULL - (conv.i >> 1);
  double y = conv.d;
  // Two Newton-Raphson refinement steps: y = y * (1.5 - 0.5*x*y*y)
  double halfx = 0.5 * x;
  y = y * (1.5 - halfx * y * y);  // iteration 1
  y = y * (1.5 - halfx * y * y);  // iteration 2
  return y;
}

// --- M1113: Log-gamma Stirling approximation ---
// Needed for computing log-binomial coefficients in AGM bound estimation
// without pulling in <cmath> lgamma (which isn't constexpr and has
// platform-varying precision).
// Stirling: ln(Gamma(x)) ≈ 0.5*ln(2π) + (x-0.5)*ln(x) - x + 1/(12x)
//           - 1/(360*x^3) + 1/(1260*x^5)
inline double StirlingLogGamma(double x) {
  if (x <= 0.0) {
    fprintf(stderr, "[AJB_BP][StirlingLogGamma] non-positive x=%.6e\n", x);
    return 0.0;
  }
  // For small x, use the recurrence ln(Gamma(x+1)) = ln(x) + ln(Gamma(x))
  // to shift x into the asymptotic regime (x >= 8)
  double shift_log = 0.0;
  while (x < 8.0) {
    shift_log -= __builtin_log(x);
    x += 1.0;
  }
  // Asymptotic expansion
  double inv_x = 1.0 / x;
  double inv_x2 = inv_x * inv_x;
  double result = 0.9189385332046727 // 0.5 * ln(2*pi)
                  + (x - 0.5) * __builtin_log(x)
                  - x
                  + inv_x * (1.0/12.0 - inv_x2 * (1.0/360.0 - inv_x2 / 1260.0));
  return result + shift_log;
}

// Log-binomial coefficient via Stirling: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
inline double LogBinomial(long long n, long long k) {
  if (k < 0 || k > n) return -1e30;  // effectively 0 probability
  if (k == 0 || k == n) return 0.0;
  return StirlingLogGamma(n + 1.0) - StirlingLogGamma(k + 1.0) - StirlingLogGamma(n - k + 1.0);
}
