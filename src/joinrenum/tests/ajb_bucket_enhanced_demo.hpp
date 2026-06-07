#ifndef JOINRENUM_BUCKET_HPP
#define JOINRENUM_BUCKET_HPP

///////////////////////////////////////////////////////////////////////////////
//  Bucket.hpp  —  joinrenum (CPU Join Enumeration Algorithm Library)
//
//  Enhanced with:
//    1. CountingBloomFilter  (MurmurHash3-based, auto-updated on insert)
//    2. reservoir_sampling   (Vitter's Algorithm R)
//    3. cache_partition       (Morton Z-curve, cache-line-aligned partitions)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace joinrenum {

// ===================================================================
//  MurmurHash3  —  finalization-mix + 128-bit variant for two hashes
//  (public-domain reference implementation by Austin Appleby)
// ===================================================================
namespace detail {

inline uint64_t murmur3_fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/// Produce two independent 64-bit hashes from a single 64-bit key.
/// Uses the 128-bit MurmurHash3 mixing strategy: the key is mixed with
/// two different constants, then each half is finalised independently.
inline std::pair<uint64_t, uint64_t> murmur3_hash128(uint64_t key,
                                                      uint64_t seed = 0) {
    // Combine key with seed using two different mix constants
    // (simulates the body rounds of MurmurHash3_x64_128).
    uint64_t h1 = seed ^ (key * 0x87c37b91114253d5ULL);
    uint64_t h2 = seed ^ (key * 0x4cf5ad432745937fULL);

    // Mix h1 and h2 together (as in the tail of MurmurHash3).
    h1 += h2;
    h2 += h1;

    // Final avalanche.
    h1 = murmur3_fmix64(h1);
    h2 = murmur3_fmix64(h2);

    h1 += h2;
    h2 += h1;

    return {h1, h2};
}

/// Generate the i-th hash using the Kirsch-Mitzenmacker double-hashing
/// scheme:  g_i(x) = h1(x) + i * h2(x)   (mod m).
inline uint64_t murmur3_nth_hash(uint64_t h1, uint64_t h2,
                                  uint32_t i, uint64_t filter_size) {
    return (h1 + static_cast<uint64_t>(i) * h2) % filter_size;
}

} // namespace detail

// ===================================================================
//  1.  CountingBloomFilter
//
//      • 4-bit counters (nibble-packed) → max count 15 per slot.
//      • k independent hash functions via Kirsch-Mitzenmacker scheme
//        on top of MurmurHash3-128.
//      • insert()   – increments counters (saturates at 15).
//      • remove()   – decrements counters (floors at 0).
//      • contains() – returns true if ALL k counters > 0.
//      • Debug prints on every operation.
// ===================================================================
class CountingBloomFilter {
public:
    /// @param num_bits   Number of logical counter slots (m).
    /// @param num_hashes Number of hash functions       (k).
    /// @param seed       Seed forwarded to MurmurHash3.
    explicit CountingBloomFilter(std::size_t num_bits   = 1024,
                                 std::size_t num_hashes = 4,
                                 uint64_t    seed       = 42)
        : m_(num_bits),
          k_(num_hashes),
          seed_(seed),
          // Each byte holds two 4-bit counters → ceil(m / 2) bytes.
          counters_((num_bits + 1) / 2, 0)
    {
        std::cout << "[BloomFilter] Constructed: m=" << m_
                  << " k=" << k_ << " seed=" << seed_
                  << " storage=" << counters_.size() << " bytes\n";
    }

    // -- Accessors -------------------------------------------------------
    std::size_t num_bits()   const noexcept { return m_; }
    std::size_t num_hashes() const noexcept { return k_; }

    // -- Core operations -------------------------------------------------

    void insert(uint64_t key) {
        auto [h1, h2] = detail::murmur3_hash128(key, seed_);
        std::cout << "[BloomFilter::insert] key=" << key
                  << "  h1=0x" << std::hex << h1
                  << " h2=0x" << h2 << std::dec << "  slots={";
        for (std::size_t i = 0; i < k_; ++i) {
            uint64_t idx = detail::murmur3_nth_hash(h1, h2,
                               static_cast<uint32_t>(i), m_);
            increment(idx);
            std::cout << idx;
            if (i + 1 < k_) std::cout << ", ";
        }
        std::cout << "}\n";
    }

    void remove(uint64_t key) {
        auto [h1, h2] = detail::murmur3_hash128(key, seed_);
        std::cout << "[BloomFilter::remove] key=" << key << "  slots={";
        for (std::size_t i = 0; i < k_; ++i) {
            uint64_t idx = detail::murmur3_nth_hash(h1, h2,
                               static_cast<uint32_t>(i), m_);
            decrement(idx);
            std::cout << idx;
            if (i + 1 < k_) std::cout << ", ";
        }
        std::cout << "}\n";
    }

    /// Fast negative lookup.  Returns false → key is definitely absent.
    bool contains(uint64_t key) const {
        auto [h1, h2] = detail::murmur3_hash128(key, seed_);
        bool result = true;
        std::cout << "[BloomFilter::contains] key=" << key << "  counters={";
        for (std::size_t i = 0; i < k_; ++i) {
            uint64_t idx = detail::murmur3_nth_hash(h1, h2,
                               static_cast<uint32_t>(i), m_);
            uint8_t c = get_counter(idx);
            std::cout << static_cast<int>(c);
            if (c == 0) result = false;
            if (i + 1 < k_) std::cout << ", ";
        }
        std::cout << "} → " << (result ? "MAYBE_PRESENT" : "DEFINITELY_ABSENT")
                  << "\n";
        return result;
    }

    void clear() {
        std::fill(counters_.begin(), counters_.end(), 0);
        std::cout << "[BloomFilter::clear] All counters reset.\n";
    }

    /// Print a human-readable dump of all non-zero counters.
    void dump() const {
        std::cout << "[BloomFilter::dump] Non-zero counters:\n";
        for (std::size_t i = 0; i < m_; ++i) {
            uint8_t c = get_counter(i);
            if (c > 0)
                std::cout << "  slot[" << i << "] = " << static_cast<int>(c) << "\n";
        }
    }

private:
    std::size_t          m_;        // number of counter slots
    std::size_t          k_;        // number of hash functions
    uint64_t             seed_;
    std::vector<uint8_t> counters_; // nibble-packed 4-bit counters

    uint8_t get_counter(std::size_t idx) const {
        uint8_t byte = counters_[idx / 2];
        return (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    }

    void set_counter(std::size_t idx, uint8_t val) {
        uint8_t& byte = counters_[idx / 2];
        if (idx % 2 == 0) {
            byte = (byte & 0xF0) | (val & 0x0F);
        } else {
            byte = (byte & 0x0F) | ((val & 0x0F) << 4);
        }
    }

    void increment(std::size_t idx) {
        uint8_t c = get_counter(idx);
        if (c < 15) set_counter(idx, c + 1); // saturate at 15
    }

    void decrement(std::size_t idx) {
        uint8_t c = get_counter(idx);
        if (c > 0) set_counter(idx, c - 1);
    }
};

// ===================================================================
//  2 & 3.  Bucket<Key>  —  the core hash-bucket container
//
//      Stores elements (join keys / tuple identifiers) in a flat vector.
//      Augmented with:
//        • CountingBloomFilter for fast negative membership tests.
//        • reservoir_sampling()  (Vitter's Algorithm R).
//        • cache_partition()     (Morton Z-curve + cache-line alignment).
// ===================================================================

/// @tparam Key  Integral key type (uint32_t, uint64_t, …).
template <typename Key = uint64_t>
class Bucket {
    static_assert(std::is_integral<Key>::value,
                  "Bucket<Key>: Key must be an integral type.");

public:
    // -- Types -----------------------------------------------------------
    using key_type       = Key;
    using container_type = std::vector<Key>;
    using iterator       = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    // -- Construction ----------------------------------------------------

    /// @param bf_num_bits    Bloom filter slot count (m).
    /// @param bf_num_hashes  Bloom filter hash count (k).
    /// @param bf_seed        Bloom filter MurmurHash3 seed.
    explicit Bucket(std::size_t bf_num_bits   = 2048,
                    std::size_t bf_num_hashes = 5,
                    uint64_t    bf_seed       = 42)
        : bloom_filter_(bf_num_bits, bf_num_hashes, bf_seed)
    {
        std::cout << "[Bucket] Constructed. Bloom filter: m="
                  << bf_num_bits << " k=" << bf_num_hashes << "\n";
    }

    // -- Element access --------------------------------------------------

    std::size_t       size()  const noexcept { return data_.size(); }
    bool              empty() const noexcept { return data_.empty(); }
    const Key&        operator[](std::size_t i) const { return data_[i]; }
    Key&              operator[](std::size_t i)       { return data_[i]; }
    iterator          begin()       noexcept { return data_.begin(); }
    iterator          end()         noexcept { return data_.end(); }
    const_iterator    begin() const noexcept { return data_.begin(); }
    const_iterator    end()   const noexcept { return data_.end(); }
    const container_type& data() const noexcept { return data_; }

    // -- Bloom-filter-accelerated insert & lookup ------------------------

    /// Insert a key.  The bloom filter is updated automatically.
    void insert(const Key& key) {
        std::cout << "[Bucket::insert] key=" << key << "\n";
        data_.push_back(key);
        bloom_filter_.insert(static_cast<uint64_t>(key));
    }

    /// Bulk insert from an iterator range.
    template <typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first)
            insert(*first);
    }

    /// Fast membership test.  A false return is *definitive* (the key is
    /// not in the bucket).  A true return means "probably present" — the
    /// caller should fall through to a linear scan for certainty.
    bool contains(const Key& key) const {
        std::cout << "[Bucket::contains] key=" << key << "\n";

        // Phase 1: Bloom filter fast path.
        if (!bloom_filter_.contains(static_cast<uint64_t>(key))) {
            std::cout << "  → Bloom filter says DEFINITELY ABSENT. Skipping scan.\n";
            return false;
        }

        // Phase 2: Linear scan for confirmation.
        std::cout << "  → Bloom filter says MAYBE PRESENT. Scanning "
                  << data_.size() << " elements…\n";
        for (std::size_t i = 0; i < data_.size(); ++i) {
            if (data_[i] == key) {
                std::cout << "  → FOUND at position " << i << ".\n";
                return true;
            }
        }
        std::cout << "  → FALSE POSITIVE — not found after full scan.\n";
        return false;
    }

    /// Remove the first occurrence of @p key.
    /// Updates the bloom filter (counting filter supports removal).
    bool remove(const Key& key) {
        auto it = std::find(data_.begin(), data_.end(), key);
        if (it == data_.end()) return false;
        data_.erase(it);
        bloom_filter_.remove(static_cast<uint64_t>(key));
        std::cout << "[Bucket::remove] key=" << key << " removed.\n";
        return true;
    }

    void clear() {
        data_.clear();
        bloom_filter_.clear();
        std::cout << "[Bucket::clear] Bucket emptied.\n";
    }

    /// Direct access to the bloom filter (e.g. for diagnostics).
    const CountingBloomFilter& bloom_filter() const noexcept {
        return bloom_filter_;
    }

    // ================================================================
    //  ALGORITHM 2:  Reservoir Sampling  —  Vitter's Algorithm R
    //
    //  Uniformly sample k elements from the bucket without knowing n
    //  ahead of time (although here n = size(); the implementation is
    //  the canonical streaming form so it generalises trivially).
    //
    //  Reference:
    //    J. S. Vitter, "Random Sampling with a Reservoir",
    //    ACM Trans. Math. Softw. 11(1), 1985, pp. 37-57.
    // ================================================================

    /// @param k       Desired sample size.
    /// @param seed    PRNG seed (0 = use std::random_device).
    /// @return        A vector of at most k uniformly sampled keys.
    std::vector<Key> reservoir_sampling(std::size_t k,
                                        uint64_t seed = 0) const
    {
        const std::size_t n = data_.size();
        std::cout << "\n╔══════════════════════════════════════════════╗\n"
                  <<   "║  Reservoir Sampling  (Vitter's Algorithm R)  ║\n"
                  <<   "╠══════════════════════════════════════════════╣\n"
                  <<   "║  n = " << std::setw(8) << n
                  <<   "    k = " << std::setw(8) << k
                  <<   "               ║\n"
                  <<   "╚══════════════════════════════════════════════╝\n";

        if (k == 0) {
            std::cout << "[reservoir] k=0 → returning empty sample.\n";
            return {};
        }
        if (n == 0) {
            std::cout << "[reservoir] Bucket is empty → returning empty sample.\n";
            return {};
        }

        // Seed the PRNG.
        std::mt19937_64 rng;
        if (seed != 0) {
            rng.seed(seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
        }

        // --- Phase 1: Fill the reservoir with the first min(k, n) items ---
        const std::size_t fill = std::min(k, n);
        std::vector<Key> reservoir(data_.begin(), data_.begin() + fill);

        std::cout << "[reservoir] Phase 1 — filled reservoir with first "
                  << fill << " elements: {";
        for (std::size_t i = 0; i < fill; ++i) {
            std::cout << reservoir[i];
            if (i + 1 < fill) std::cout << ", ";
        }
        std::cout << "}\n";

        if (n <= k) {
            std::cout << "[reservoir] n <= k → entire bucket is the sample.\n";
            return reservoir;
        }

        // --- Phase 2: For each remaining element, replace with
        //              probability k / (i + 1) ---
        std::size_t replace_count = 0;
        for (std::size_t i = k; i < n; ++i) {
            std::uniform_int_distribution<std::size_t> dist(0, i);
            std::size_t j = dist(rng);

            if (j < k) {
                std::cout << "[reservoir] i=" << i
                          << "  element=" << data_[i]
                          << "  j=" << j << " < k=" << k
                          << "  → REPLACE reservoir[" << j << "]="
                          << reservoir[j] << " with " << data_[i] << "\n";
                reservoir[j] = data_[i];
                ++replace_count;
            } else {
                std::cout << "[reservoir] i=" << i
                          << "  element=" << data_[i]
                          << "  j=" << j << " >= k=" << k
                          << "  → SKIP\n";
            }
        }

        std::cout << "[reservoir] Done. Total replacements: "
                  << replace_count << "/" << (n - k)
                  << " candidates examined.\n";
        std::cout << "[reservoir] Final sample: {";
        for (std::size_t i = 0; i < reservoir.size(); ++i) {
            std::cout << reservoir[i];
            if (i + 1 < reservoir.size()) std::cout << ", ";
        }
        std::cout << "}\n\n";

        return reservoir;
    }

    // ================================================================
    //  ALGORITHM 3:  Cache-Aware Partitioning via Morton Z-Curve
    //
    //  Reorders the bucket's elements along a Morton (Z-order) curve
    //  and then splits them into contiguous partitions aligned to a
    //  configurable cache-line size.  The Z-order preserves 2-D spatial
    //  locality when the key is treated as a pair (high-bits, low-bits),
    //  which maps naturally to composite join keys.
    //
    //  The output partitions are cache-line-aligned in element count:
    //    partition_size = cache_line_bytes / sizeof(Key)
    //  guaranteeing that each partition starts on a fresh cache line
    //  (assuming the underlying vector storage is itself aligned, which
    //  standard allocators provide for small types).
    // ================================================================

    struct Partition {
        std::size_t start;        // inclusive index into data_
        std::size_t end;          // exclusive
        Key         min_key;
        Key         max_key;
        uint64_t    min_morton;
        uint64_t    max_morton;
    };

    /// @param cache_line_bytes  Target cache-line width (default 64 B).
    /// @return  Vector of Partition descriptors.
    std::vector<Partition> cache_partition(
            std::size_t cache_line_bytes = 64) const
    {
        std::cout << "\n┌─────────────────────────────────────────────────┐\n"
                  <<   "│  Cache-Aware Partitioning  (Morton Z-Curve)      │\n"
                  <<   "├─────────────────────────────────────────────────┤\n"
                  <<   "│  elements     = " << std::setw(10) << data_.size()
                  <<   "                      │\n"
                  <<   "│  sizeof(Key)  = " << std::setw(10) << sizeof(Key)
                  <<   "                      │\n"
                  <<   "│  cache line   = " << std::setw(10) << cache_line_bytes
                  <<   " bytes                │\n"
                  <<   "└─────────────────────────────────────────────────┘\n";

        std::vector<Partition> result;

        if (data_.empty()) {
            std::cout << "[cache_partition] Bucket is empty → 0 partitions.\n";
            return result;
        }

        // --- Step 1: Compute Morton codes for every element.
        //     We split each key into two halves (high / low bits) and
        //     interleave them to form the Z-curve index.
        const std::size_t n = data_.size();
        constexpr std::size_t half_bits = sizeof(Key) * 4; // half the key width

        struct MortonEntry {
            Key      key;
            uint64_t morton;
        };

        std::vector<MortonEntry> entries(n);
        for (std::size_t i = 0; i < n; ++i) {
            uint64_t k = static_cast<uint64_t>(data_[i]);
            // Split into two halves.
            uint64_t x = k >> half_bits;           // high half
            uint64_t y = k & ((1ULL << half_bits) - 1); // low half
            entries[i] = { data_[i], morton_interleave(x, y) };
        }

        std::cout << "[cache_partition] Morton codes computed. Examples:\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(n, 8); ++i) {
            std::cout << "  key=" << std::setw(12) << entries[i].key
                      << "  morton=0x" << std::hex << std::setw(16)
                      << std::setfill('0') << entries[i].morton
                      << std::dec << std::setfill(' ') << "\n";
        }
        if (n > 8) std::cout << "  … (" << (n - 8) << " more)\n";

        // --- Step 2: Sort by Morton code.
        std::sort(entries.begin(), entries.end(),
                  [](const MortonEntry& a, const MortonEntry& b) {
                      return a.morton < b.morton;
                  });

        std::cout << "[cache_partition] Sorted by Morton code.\n";

        // --- Step 3: Build cache-line-aligned partitions.
        std::size_t elems_per_line = cache_line_bytes / sizeof(Key);
        if (elems_per_line == 0) elems_per_line = 1;

        std::cout << "[cache_partition] Elements per cache line: "
                  << elems_per_line << "\n";

        // Write sorted order back into a result vector (we don't mutate
        // data_ because this is a const method — caller can copy out).
        std::vector<Key> sorted_keys(n);
        for (std::size_t i = 0; i < n; ++i)
            sorted_keys[i] = entries[i].key;

        std::size_t part_id = 0;
        for (std::size_t start = 0; start < n; start += elems_per_line) {
            std::size_t end = std::min(start + elems_per_line, n);
            Partition p;
            p.start       = start;
            p.end         = end;
            p.min_key     = sorted_keys[start];
            p.max_key     = sorted_keys[end - 1];
            p.min_morton  = entries[start].morton;
            p.max_morton  = entries[end - 1].morton;
            result.push_back(p);

            std::cout << "[cache_partition] Partition " << std::setw(4) << part_id
                      << ": indices [" << std::setw(6) << start
                      << ", " << std::setw(6) << end << ")"
                      << "  keys [" << std::setw(12) << p.min_key
                      << " .. " << std::setw(12) << p.max_key << "]"
                      << "  morton [0x" << std::hex
                      << std::setw(16) << std::setfill('0') << p.min_morton
                      << " .. 0x"
                      << std::setw(16) << std::setfill('0') << p.max_morton
                      << "]" << std::dec << std::setfill(' ')
                      << "  #elems=" << (end - start) << "\n";
            ++part_id;
        }

        std::cout << "[cache_partition] Total partitions: "
                  << result.size() << "\n\n";

        return result;
    }

    /// Non-const variant: actually reorders data_ in place and returns
    /// partition descriptors referencing the reordered data_.
    std::vector<Partition> cache_partition_inplace(
            std::size_t cache_line_bytes = 64)
    {
        std::cout << "[cache_partition_inplace] Reordering bucket in place.\n";

        const std::size_t n = data_.size();
        if (n == 0) return {};

        constexpr std::size_t half_bits = sizeof(Key) * 4;

        struct MortonEntry {
            Key      key;
            uint64_t morton;
        };

        std::vector<MortonEntry> entries(n);
        for (std::size_t i = 0; i < n; ++i) {
            uint64_t k = static_cast<uint64_t>(data_[i]);
            uint64_t x = k >> half_bits;
            uint64_t y = k & ((1ULL << half_bits) - 1);
            entries[i] = { data_[i], morton_interleave(x, y) };
        }

        std::sort(entries.begin(), entries.end(),
                  [](const MortonEntry& a, const MortonEntry& b) {
                      return a.morton < b.morton;
                  });

        for (std::size_t i = 0; i < n; ++i)
            data_[i] = entries[i].key;

        // Rebuild bloom filter to match reordered data.
        bloom_filter_.clear();
        for (const auto& key : data_)
            bloom_filter_.insert(static_cast<uint64_t>(key));

        // Delegate partition computation to the const version.
        return cache_partition(cache_line_bytes);
    }

    // -- Diagnostics -----------------------------------------------------

    void print(const std::string& label = "Bucket") const {
        std::cout << "[" << label << "] size=" << data_.size() << " → {";
        for (std::size_t i = 0; i < data_.size(); ++i) {
            std::cout << data_[i];
            if (i + 1 < data_.size()) std::cout << ", ";
        }
        std::cout << "}\n";
    }

private:
    container_type     data_;
    CountingBloomFilter bloom_filter_;

    // ----------------------------------------------------------------
    //  Morton (Z-order) bit interleaving
    //
    //  Interleave the lowest 32 bits of x and y into a single 64-bit
    //  value:  result[2i]   = x[i]
    //          result[2i+1] = y[i]
    //  Uses the standard bit-spreading technique with magic constants.
    // ----------------------------------------------------------------

    static uint64_t spread_bits_32(uint64_t v) {
        // Spread the lower 32 bits of v so that each original bit sits
        // at an even position:   abcd… → .a.b.c.d…
        v &= 0x00000000FFFFFFFFULL;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v <<  8)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v <<  2)) & 0x3333333333333333ULL;
        v = (v | (v <<  1)) & 0x5555555555555555ULL;
        return v;
    }

    static uint64_t morton_interleave(uint64_t x, uint64_t y) {
        return spread_bits_32(x) | (spread_bits_32(y) << 1);
    }
};

} // namespace joinrenum

#endif // JOINRENUM_BUCKET_HPP
