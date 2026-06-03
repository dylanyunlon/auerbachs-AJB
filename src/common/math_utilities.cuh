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
