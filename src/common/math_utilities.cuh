#pragma once
// [AJB] DivideUp/RoundUp: 所有chunk大小计算和对齐的基础, _KB/_MB/_GB是allocator用的字面量

constexpr unsigned long long operator"" _KB(unsigned long long n) { return n * 1024; }
constexpr unsigned long long operator"" _MB(unsigned long long n) { return n * 1024 * 1024; }
constexpr unsigned long long operator"" _GB(unsigned long long n) { return n * 1024 * 1024 * 1024; }

inline unsigned long long DivideUp(unsigned long long dividend, unsigned long long divisor) {
  return (dividend + divisor - 1) / divisor;
}

inline unsigned long long RoundUp(unsigned long long dividend, unsigned long long divisor) {
  return DivideUp(dividend, divisor) * divisor;
}
