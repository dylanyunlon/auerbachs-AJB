#pragma once
#include "common/ajb_debug_infra.cuh"

// AJB: estimate distinct values via sampling (avoids O(n) unique)
template<typename T>
static inline size_t ajb_estimate_cardinality(const T* data, size_t n,
                                               size_t sample_size = 1000) {
    if (n <= sample_size) {
        std::vector<T> tmp(data, data + n);
        std::sort(tmp.begin(), tmp.end());
        return std::unique(tmp.begin(), tmp.end()) - tmp.begin();
    }
    // Reservoir sample then count unique
    std::vector<T> sample(sample_size);
    for (size_t i = 0; i < sample_size; ++i) sample[i] = data[i];
    // LCG sampling for indices beyond sample_size
    uint64_t lcg = 6364136223846793005ULL;
    for (size_t i = sample_size; i < n; ++i) {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = lcg % (i + 1);
        if (j < sample_size) sample[j] = data[i];
    }
    std::sort(sample.begin(), sample.end());
    size_t uniq = std::unique(sample.begin(), sample.end()) - sample.begin();
    // Extrapolate: unique_est = n * (sample_unique / sample_size)
    return static_cast<size_t>((double)uniq / sample_size * n);
}

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

    // Upstream: the sigma<100 and sigma>=100 branches share a common
    // pattern: generate S keys by copying from R with modular indexing,
    // then shuffle.  The only difference is that sigma<100 first makes
    // R keys unique (even values) and then corrupts some S keys.
    // Changed: extract the shared "derive S from R" logic into a lambda.
    auto derive_s_keys_from_r = [&](size_t s_size, size_t r_size, size_t threads) {
      if (r_size == 0) return;
#pragma omp parallel for num_threads(threads)
      for (size_t i = 0; i < s_size; ++i) {
        s_relation.GetKeys()[i] = r_relation.GetKeys()[i % r_size];
      }
      __gnu_parallel::random_shuffle(s_relation.GetKeys().begin(), s_relation.GetKeys().end(),
                                     __gnu_parallel::_RandomNumber(random_seed));
    };

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

      derive_s_keys_from_r(s_relation.GetSize(), r_relation.GetSize(), num_threads);

      num_matches = (static_cast<double>(sigma) / 100) * s_relation.GetSize();
#pragma omp parallel for num_threads(num_threads)
      for (size_t i = num_matches; i < s_relation.GetSize(); ++i) {
        ++s_relation.GetKeys()[i];
      }
      // Re-shuffle after corruption
      __gnu_parallel::random_shuffle(s_relation.GetKeys().begin(), s_relation.GetKeys().end(),
                                     __gnu_parallel::_RandomNumber(random_seed));
    } else {
      derive_s_keys_from_r(s_relation.GetSize(), r_relation.GetSize(), num_threads);
    }
    DataGenerator::ComputeDistribution<V>(s_relation.GetValues().data(), s_relation.GetSize(), num_threads,
                                          value_distribution, random_seed >> 1);

    if (r_sort) {
      ParallelSortPairs(r_relation.GetKeys(), r_relation.GetValues());
    }

    if (s_sort) {
      ParallelSortPairs(s_relation.GetKeys(), s_relation.GetValues());
    }

    return num_matches;
  }
};
