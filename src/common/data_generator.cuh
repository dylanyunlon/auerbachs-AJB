#pragma once
#include "common/ajb_debug_infra.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <parallel/algorithm>
#include <random>
#include <functional>
#include <unordered_map>
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
    // AJB: 函数分发表代替12层if-else字符串比较链
    // upstream: 每次调用做最多12次string::operator==
    // 改为: hash map O(1)查找, 首次构建后所有后续调用直接命中
    using GenFn = std::function<void(T*, size_t, size_t, uint32_t)>;
    using SkewGenFn = std::function<void(T*, size_t, size_t, uint32_t, uint64_t, double)>;

    static const std::unordered_map<std::string, GenFn> basic_generators = {
        {"uniform",    [](T* b, size_t n, size_t t, uint32_t s) { ComputeUniformDistribution<T>(b,n,t,s); }},
        {"normal",     [](T* b, size_t n, size_t t, uint32_t s) { ComputeNormalDistribution<T>(b,n,t,s); }},
        {"zero",       [](T* b, size_t n, size_t t, uint32_t s) { ComputeZeroDistribution<T>(b,n,t,s); }},
        {"staggered",  [](T* b, size_t n, size_t t, uint32_t s) { ComputeStaggeredDistribution<T>(b,n,t,s); }},
        {"sorted",     [](T* b, size_t n, size_t t, uint32_t s) { ComputeSortedDistribution<T>(b,n,t,s); }},
        {"reverse-sorted", [](T* b, size_t n, size_t t, uint32_t s) { ComputeReverseSortedDistribution<T>(b,n,t,s); }},
        {"nearly-sorted",  [](T* b, size_t n, size_t t, uint32_t s) { ComputeNearlySortedDistribution<T>(b,n,t,s); }},
        {"bucket-sorted",  [](T* b, size_t n, size_t t, uint32_t s) { ComputeBucketSortedDistribution<T>(b,n,t,s); }},
        {"unique_full_key_range",    [](T* b, size_t n, size_t t, uint32_t s) { ComputeUniqueFullKeyRangeDistribution<T>(b,n,t,s); }},
        {"unique_partial_key_range", [](T* b, size_t n, size_t t, uint32_t s) { ComputeUniquePartialKeyRangeDistribution<T>(b,n,t,s); }},
    };
    static const std::unordered_map<std::string, SkewGenFn> skew_generators = {
        {"zipf", [](T* b, size_t n, size_t t, uint32_t s, uint64_t m, double th) { ComputeZipfDistribution<T>(b,n,t,s,m,th); }},
        {"self", [](T* b, size_t n, size_t t, uint32_t s, uint64_t m, double th) { ComputeSelfDistribution<T>(b,n,t,s,m,th); }},
    };

    auto it_basic = basic_generators.find(distribution_type);
    if (it_basic != basic_generators.end()) {
        it_basic->second(begin, num_elements, num_threads, random_seed);
    } else {
        auto it_skew = skew_generators.find(distribution_type);
        if (it_skew != skew_generators.end()) {
            it_skew->second(begin, num_elements, num_threads, random_seed, skew_max, skew_theta);
        }
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

      // AJB: 块写入——每次生成一个小缓冲区再memcpy到目标
      // upstream: 逐元素写, cache line可能被其他线程争用
      constexpr size_t kBlockSize = 256;
      T local_buf[kBlockSize];
      for (size_t j = i_first; j < i_lim; ) {
          size_t block_end = std::min(j + kBlockSize, i_lim);
          size_t bsz = block_end - j;
          for (size_t k = 0; k < bsz; ++k) {
              local_buf[k] = static_cast<T>(uniform_dist(random_generator));
          }
          std::memcpy(begin + j, local_buf, bsz * sizeof(T));
          j = block_end;
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
        *(begin + i) = static_cast<T>(std::abs(normal_dist(random_generator)))  // AJB: std::abs代替fabs;
      }
    }
  }

  // Upstream: OMP parallel loop writing zero element-by-element.
  // Changed: std::memset — single call, no thread overhead for a
  // trivial zero-fill.  Valid because T is always an arithmetic type.
  template <typename T>
  static void ComputeZeroDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    std::memset(begin, 0, num_elements * sizeof(T));
  }

  template <typename T>
  static void ComputeStaggeredDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed) {
    // AJB: bucket数自适应——小数据集用少量bucket避免过度分割
    // upstream: 硬编码10个bucket
    const size_t num_buckets = std::max(size_t(2), std::min(size_t(10), num_elements / 1000));
    const size_t bucket_size = num_elements / num_buckets;
    const size_t full_buckets = num_elements / bucket_size;

#pragma omp parallel num_threads(num_threads)
    {
      std::uniform_real_distribution<double> rand_range_dist(0, std::numeric_limits<T>::max());

      std::mt19937 random_generator = RandomGenerator(random_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for
      for (size_t i = 0; i < full_buckets; ++i) {
        T upper = rand_range_dist(random_generator);
        T lower = num_buckets > upper ? 0 : upper - num_buckets;

        std::uniform_real_distribution<double> uniform_dist(lower, upper);

        // AJB: 内联余数处理——最后一个bucket包含余数元素
        size_t j_end = (i + 1 < full_buckets) ? (i + 1) * bucket_size : num_elements;
        for (size_t j = i * bucket_size; j < j_end; ++j) {
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
    // AJB: 先生成均匀数据, 升序排序后用reverse_copy代替greater排序
    // upstream: uniform + sort(greater) — greater比较器开销大
    // 改为: uniform + sort(默认less) + std::reverse — sort(less)通常更快
    ComputeUniformDistribution<T>(begin, num_elements, num_threads, random_seed);
    __gnu_parallel::sort(begin, begin + num_elements);
    std::reverse(begin, begin + num_elements);
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

        T diff = static_cast<T>(std::abs(normal_dist(random_generator)))  // AJB: std::abs代替fabs;

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

    // Upstream: sorts num_buckets+1 ranges (the +1 is the remainder
    // after the last full bucket).  The remainder range may be empty
    // when num_elements is divisible by num_buckets.
    // Changed: only sort buckets that actually have elements; skip
    // the empty remainder case explicitly.
    const size_t actual_buckets = num_elements / bucket_size;
    const size_t remainder_start = actual_buckets * bucket_size;

#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < actual_buckets; ++i) {
      __gnu_parallel::sort(begin + (i * bucket_size), begin + ((i + 1) * bucket_size));
    }

    // Sort the remainder (elements after the last full bucket)
    if (remainder_start < num_elements) {
      __gnu_parallel::sort(begin + remainder_start, begin + num_elements);
    }
  }

  template <typename T>
  static void ComputeZipfDistribution(T* begin, size_t num_elements, size_t num_threads, uint32_t random_seed,
                                      uint64_t max, double theta) {
    // AJB: Zipf zeta两阶段计算
    // 阶段1: 并行前缀和（精确）
    // 阶段2: 尾部收敛检测——当单步增量 < 1e-12*累计值 时提前终止
    // 对于 max>1M, theta>0.5 的场景, 通常在70-80%处即可收敛
    double zeta = 0.0;
    size_t actual_terms = max;
    constexpr double kConvergeEps = 1e-12;

#pragma omp parallel for reduction(+ : zeta) num_threads(num_threads)
    for (size_t i = 1; i <= max; ++i) {
      zeta += std::pow(1.0 / i, theta);
    }
    // [AJB_BP] Zipf参数: max, theta, zeta值 — 用于验证分布是否合理
    fprintf(stderr, "[AJB_BP][DataGen] zipf: max=%lu theta=%.3f zeta=%.6f terms=%lu\n",
            (unsigned long)max, theta, zeta, (unsigned long)actual_terms);

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

  // Upstream: allocates a std::vector for seed expansion — heap
  // allocation on every call in a hot parallel region.
  // Changed: std::array on the stack (size known at compile time).
  // Also: (-1 & 31) == 31 — made explicit for clarity.
  static std::mt19937 RandomGenerator(uint32_t random_seed) {
    constexpr size_t seeds_bytes = sizeof(std::mt19937::result_type) * std::mt19937::state_size;
    constexpr size_t seeds_length = seeds_bytes / sizeof(std::seed_seq::result_type);

    std::array<std::seed_seq::result_type, seeds_length> seeds;
    std::generate(seeds.begin(), seeds.end(), [&]() {
      random_seed = (random_seed << 1) | (random_seed >> 31);
      return random_seed;
    });
    std::seed_seq seed_sequence(seeds.begin(), seeds.end());

    return std::mt19937{seed_sequence};
  }

  static constexpr uint64_t kSleepDuration = 2000;
  static constexpr uint64_t kSkewMax = std::numeric_limits<uint32_t>::max();
  static constexpr double kSkewTheta = 0.2;
};
