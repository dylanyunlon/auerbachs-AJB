#pragma once

#include <string>

#include <parallel/random_shuffle.h>

#include "data_generator.cuh"
#include "parallel_algorithms.cuh"
#include "pinned_vector.cuh"

template <typename T, typename V>
class Relation {
 public:
  explicit Relation(size_t num_elements) : size_(num_elements), keys_(num_elements), values_(num_elements) {}

  size_t GetSize() const { return size_; }

  PinnedVector<T>& GetKeys() { return keys_; };

  PinnedVector<V>& GetValues() { return values_; };

 private:
  const size_t size_;

  PinnedVector<T> keys_;
  PinnedVector<V> values_;
};

class RelationGenerator {
 public:
  template <typename T, typename V>
  static size_t ComputeDistributions(Relation<T, V>& r_relation, Relation<T, V>& s_relation, size_t num_threads,
                                     const std::string& key_distribution, const std::string& value_distribution,
                                     uint32_t random_seed, uint32_t theta, uint32_t sigma, bool r_sort, bool s_sort) {
    DataGenerator::ComputeDistribution<T>(r_relation.GetKeys().data(), r_relation.GetSize(), num_threads,
                                          key_distribution, random_seed);
    DataGenerator::ComputeDistribution<V>(r_relation.GetValues().data(), r_relation.GetSize(), num_threads,
                                          value_distribution, random_seed << 1);

    size_t num_matches = s_relation.GetSize();

    if (theta > 0) {
      const uint64_t skew_max = r_relation.GetSize() - 1;
      const double skew_theta = static_cast<double>(theta) / 100;

      DataGenerator::ComputeDistribution<T>(s_relation.GetKeys().data(), s_relation.GetSize(), num_threads, "zipf",
                                            random_seed, skew_max, skew_theta);
    } else if (sigma < 100) {
#pragma omp parallel for num_threads(num_threads)
      for (size_t i = 0; i < r_relation.GetSize(); ++i) {
        r_relation.GetKeys()[i] = i * 2;
      }
      __gnu_parallel::random_shuffle(r_relation.GetKeys().begin(), r_relation.GetKeys().end(),
                                     __gnu_parallel::_RandomNumber(random_seed));

#pragma omp parallel for num_threads(num_threads)
      for (size_t i = 0; i < s_relation.GetSize(); ++i) {
        s_relation.GetKeys()[i] = r_relation.GetKeys()[i % r_relation.GetSize()];
      }
      num_matches = (static_cast<double>(sigma) / 100) * s_relation.GetSize();
#pragma omp parallel for num_threads(num_threads)
      for (size_t i = num_matches; i < s_relation.GetSize(); ++i) {
        ++s_relation.GetKeys()[i];
      }
      __gnu_parallel::random_shuffle(s_relation.GetKeys().begin(), s_relation.GetKeys().end(),
                                     __gnu_parallel::_RandomNumber(random_seed));
    } else {
#pragma omp parallel for num_threads(num_threads)
      for (size_t i = 0; i < s_relation.GetSize(); ++i) {
        s_relation.GetKeys()[i] = r_relation.GetKeys()[i % r_relation.GetSize()];
      }
      __gnu_parallel::random_shuffle(s_relation.GetKeys().begin(), s_relation.GetKeys().end(),
                                     __gnu_parallel::_RandomNumber(random_seed));
    }
    DataGenerator::ComputeDistribution<V>(s_relation.GetValues().data(), s_relation.GetSize(), num_threads,
                                          value_distribution, random_seed >> 1);

    if (r_sort) {
      ParallelSortPairs(r_relation.GetKeys(), r_relation.GetValues());
    }

    if (s_sort) {
      ParallelSortPairs(s_relation.GetKeys(), s_relation.GetValues());
    }

    // [AJB] expected_matches控制join output大小: theta>0用zipf偏斜,sigma<100引入non-matching keys
    fprintf(stderr, "[AJB_STATE][RelationGen] |R|=%zu |S|=%zu expected_matches=%zu theta=%u sigma=%u\n",
            r_relation.GetSize(), s_relation.GetSize(), num_matches, theta, sigma);
    return num_matches;
  }
};
