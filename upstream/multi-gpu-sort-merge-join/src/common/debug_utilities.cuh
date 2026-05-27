#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include <termcolor/termcolor.hpp>

#include "pinned_vector.cuh"

template <typename T>
bool IsSortedPrintFailures(const PinnedVector<T>& vector) {
  bool is_sorted = true;
  for (size_t i = 1; i < vector.size(); ++i) {
    if (vector[i - 1] > vector[i]) {
      const size_t begin = std::max<size_t>(0, i - 5);
      const size_t end = std::min<size_t>(i + 5, vector.size() - 1);

      std::cout << "[ERROR] IsSortedPrintFailures: ";
      for (size_t j = begin; j < end; ++j) {
        if (i == j) {
          std::cout << termcolor::red;
        }
        std::cout << vector[j];
        if (i == j) {
          std::cout << termcolor::reset;
        }
        std::cout << ", ";
      }
      std::cout << "i = " << termcolor::red << i << termcolor::reset << std::endl;
      is_sorted = false;
    }
  }
  return is_sorted;
}

template <typename T>
void PrintVector(const std::string& name, const PinnedVector<T>& vector) {
  std::cout << name << ": ";
  for (size_t i = 0; i < vector.size(); ++i) {
    std::cout << vector[i];
    if (i < vector.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;
}
