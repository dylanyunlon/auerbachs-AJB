#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <parallel/algorithm>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include <omp.h>
#include <randomdist/selfsimilar_int_distribution.h>
#include <randomdist/zipfian_int_distribution.h>

#include "math_utilities.cuh"

class DataGenerator {
 public:
  template <typename T>
  static void ComputeDistribution(T* begin, size_t num_elements, size_t num_threads,
                                  const std::string& distribution_type, uint32_t random_seed,
                                  uint64_t skew_max = kSkewMax, double skew_theta = kSkewTheta) {
    if (distribution_type == "uniform") {
      ComputeUniformDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "normal") {
      ComputeNormalDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "zero") {
      ComputeZeroDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "staggered") {
      ComputeStaggeredDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "sorted") {
      ComputeSortedDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "reverse-sorted") {
      ComputeReverseSortedDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "nearly-sorted") {
      ComputeNearlySortedDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "bucket-sorted") {
      ComputeBucketSortedDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "zipf") {
      ComputeZipfDistribution<T>(begin, num_elements, num_threads, random_seed, skew_max, skew_theta);
    } else if (distribution_type == "self") {
      ComputeSelfDistribution<T>(begin, num_elements, num_threads, random_seed, skew_max, skew_theta);
    } else if (distribution_type == "unique_full_key_range") {
      ComputeUniqueFullKeyRangeDistribution<T>(begin, num_elements, num_threads, random_seed);
    } else if (distribution_type == "unique_partial_key_range") {
      ComputeUniquePartialKeyRangeDistribution<T>(begin, num_elements, num_threads, random_seed);
    }
    // [AJB] 分布采样: 打印头/中/尾3个值和极值,验证分布参数是否生效
    if (num_elements >= 3) {
      T v0 = begin[0], vm = begin[num_elements/2], vn = begin[num_elements-1];
      fprintf(stderr, "[AJB_STATE][DataGen] dist=%s n=%zu  head=%llu mid=%llu tail=%llu  seed=%u\n",
              distribution_type.c_str(), num_elements,
              (unsigned long long)v0, (unsigned long long)vm, (unsigned long long)vn, random_seed);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepDuration));
  }

 private:
  template <typename T>
  static void ComputeUniformDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    const size_t n_per_thread = DivideUp(num_elements, num_threads);

#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_threads; ++i) {
      std::uniform_real_distribution<double> uniform_dist(0, std::numeric_limits<T>::max());

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

      const size_t i_first = n_per_thread * i;
      const size_t i_lim = std::min(n_per_thread + i_first, num_elements);

      if (i_lim <= i_first) {
        continue;
      }

      for (size_t j = i_first; j < i_lim; ++j) {
        *(begin + j) = static_cast<T>(uniform_dist(random_generator));
      }
    }
  }

  template <typename T>
  static void ComputeNormalDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    const double mean = std::numeric_limits<T>::max() / 2.0;
    const double stddev = mean / 3.0;

#pragma omp parallel num_threads(num_threads)
    {
      std::normal_distribution<double> normal_dist(mean, stddev);

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(std::fabs(normal_dist(random_generator)));
      }
    }
  }

  template <typename T>
  static void ComputeZeroDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_elements; ++i) {
      *(begin + i) = 0;
    }
  }

  template <typename T>
  static void ComputeStaggeredDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    const size_t num_buckets = 10;
    const size_t bucket_size = num_elements / num_buckets;

#pragma omp parallel num_threads(num_threads)
    {
      std::uniform_real_distribution<double> rand_range_dist(0, std::numeric_limits<T>::max());

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_elements / bucket_size; ++i) {
        T upper = rand_range_dist(random_generator);
        T lower = num_buckets > upper ? 0 : upper - num_buckets;

        std::uniform_real_distribution<double> uniform_dist(lower, upper);

        for (size_t j = i * bucket_size; j < (i + 1) * bucket_size; ++j) {
          *(begin + j) = static_cast<T>(uniform_dist(random_generator));
        }
      }
    }

    std::mt19937 random_generator = RandomGenerator(random_seed + num_threads);
    std::uniform_real_distribution<double> rand_range_dist(0, std::numeric_limits<T>::max());

    T upper = rand_range_dist(random_generator);
    T lower = num_buckets > upper ? 0 : upper - num_buckets;

    std::uniform_real_distribution<double> uniform_dist(lower, upper);

    for (size_t i = num_elements - (num_elements % bucket_size); i < num_elements; ++i) {
      *(begin + i) = static_cast<T>(uniform_dist(random_generator));
    }
  }

  template <typename T>
  static void ComputeSortedDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    ComputeUniformDistribution<T>(begin, num_elements, num_threads, random_seed);
    __gnu_parallel::sort(begin, begin + num_elements);
  }

  template <typename T>
  static void ComputeReverseSortedDistribution(T* begin, size_t num_elements, size_t num_threads,
                                               uint32_t random_seed) {
    ComputeUniformDistribution<T>(begin, num_elements, num_threads, random_seed);
    __gnu_parallel::sort(begin, begin + num_elements, std::greater<T>());
  }

  template <typename T>
  static void ComputeNearlySortedDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    ComputeSortedDistribution<T>(begin, num_elements, num_threads, random_seed);

#pragma omp parallel num_threads(num_threads)
    {
      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_elements - 1; ++i) {
        const double mean = 0.0;

        double stddev = *(begin + i + 1) - *(begin + i);

        if (stddev < std::numeric_limits<double>::max() / 2.0) {
          stddev *= 2.0;
        }

        std::normal_distribution<double> normal_dist(mean, stddev);

        T diff = static_cast<T>(std::fabs(normal_dist(random_generator)));

        if (*(begin + i) > std::numeric_limits<T>::max() - diff) {
          *(begin + i) = std::numeric_limits<T>::max();
        } else {
          *(begin + i) += diff;
        }
      }
    }
  }

  template <typename T>
  static void ComputeBucketSortedDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    const size_t num_buckets = 10;
    const size_t bucket_size = num_elements / num_buckets;

#pragma omp parallel num_threads(num_threads)
    {
      std::uniform_real_distribution<double> uniform_dist(0, std::numeric_limits<T>::max());

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_buckets; ++i) {
        for (size_t j = i * bucket_size; j < (i + 1) * bucket_size; ++j) {
          *(begin + j) = static_cast<T>(uniform_dist(random_generator));
        }
      }

#pragma omp for
      for (size_t i = num_elements - num_elements % bucket_size; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(uniform_dist(random_generator));
      }
    }

#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i <= num_buckets; ++i) {
      __gnu_parallel::sort(begin + (i * bucket_size), std::min(begin + (i + 1) * bucket_size, begin + num_elements));
    }
  }

  template <typename T>
  static void ComputeZipfDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed,
                                      uint64_t max, double theta) {
    double zeta = 0.0;
#pragma omp parallel for reduction(+ : zeta) num_threads(num_threads)
    for (size_t i = 1; i <= max; ++i) {
      zeta += std::pow(1.0 / i, theta);
    }

#pragma omp parallel num_threads(num_threads)
    {
      zipfian_int_distribution<int64_t>::param_type param_type(0, max, theta, zeta);
      zipfian_int_distribution<int64_t> zipf_dist(param_type);

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(zipf_dist(random_generator));
      }
    }
  }

  template <typename T>
  static void ComputeSelfDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed,
                                      uint64_t max, double theta) {
#pragma omp parallel num_threads(num_threads)
    {
      selfsimilar_int_distribution<int64_t>::param_type param_type(0, max, theta);
      selfsimilar_int_distribution<int64_t> self_dist(param_type);

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < num_elements; ++i) {
        *(begin + i) = static_cast<T>(self_dist(random_generator));
      }
    }
  }

  template <typename T>
  static void ComputeUniqueFullKeyRangeDistribution(T* begin, size_t num_elements, size_t num_threads,
                                                    uint32_t random_seed) {
    const size_t n_per_thread = DivideUp(num_elements, num_threads);
    const double range_per_random = std::floor(std::numeric_limits<T>::max() / num_elements);

#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_threads; ++i) {
      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

      const size_t i_first = n_per_thread * i;
      const size_t i_lim = std::min(n_per_thread + i_first, num_elements);

      if (i_lim <= i_first) {
        continue;
      }

      for (size_t j = i_first; j < i_lim; ++j) {
        const double beg_range = range_per_random * j;
        const double end_range = std::nextafter(beg_range + range_per_random, beg_range);

        std::uniform_real_distribution<double> uniform_dist(beg_range, end_range);

        begin[j] = static_cast<T>(uniform_dist(random_generator));
      }
    }
    __gnu_parallel::random_shuffle(begin, begin + num_elements, __gnu_parallel::_RandomNumber(random_seed));
  }

  template <typename T>
  static void ComputeUniquePartialKeyRangeDistribution(T* begin, size_t num_elements, size_t num_threads,
                                                       uint32_t random_seed) {
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_elements; ++i) {
      begin[i] = static_cast<T>(i);
    }
    __gnu_parallel::random_shuffle(begin, begin + num_elements, __gnu_parallel::_RandomNumber(random_seed));
  }

  static std::mt19937 RandomGenerator(uint32_t random_seed) {
    const size_t seeds_bytes = sizeof(std::mt19937::result_type) * std::mt19937::state_size;
    const size_t seeds_length = seeds_bytes / sizeof(std::seed_seq::result_type);

    std::vector<std::seed_seq::result_type> seeds(seeds_length);
    std::generate(seeds.begin(), seeds.end(), [&]() {
      random_seed = (random_seed << 1) | (random_seed >> (-1 & 31));
      return random_seed;
    });
    std::seed_seq seed_sequence(seeds.begin(), seeds.end());

    return std::mt19937{seed_sequence};
  }

  static constexpr uint64_t kSleepDuration = 2000;
  static constexpr uint64_t kSkewMax = std::numeric_limits<uint32_t>::max();
  static constexpr double kSkewTheta = 0.2;
};
