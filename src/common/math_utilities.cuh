#pragma once

constexpr unsigned long long operator"" _KB(unsigned long long n) { return n * 1024; }
constexpr unsigned long long operator"" _MB(unsigned long long n) { return n * 1024 * 1024; }
constexpr unsigned long long operator"" _GB(unsigned long long n) { return n * 1024 * 1024 * 1024; }

inline unsigned long long DivideUp(unsigned long long dividend, unsigned long long divisor) {
  return (dividend + divisor - 1) / divisor;
}

inline unsigned long long RoundUp(unsigned long long dividend, unsigned long long divisor) {
  return DivideUp(dividend, divisor) * divisor;
}

// [AJB] 数值工具诊断
#include <cstdio>
static inline void ajb_report_div_round(size_t n, size_t d, size_t result, const char* tag) {
    fprintf(stderr, "[AJB_TRACE][Math] %s: %zu / %zu = %zu (ceil)\n", tag, n, d, result);
}
