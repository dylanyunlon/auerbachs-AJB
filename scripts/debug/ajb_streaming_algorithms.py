#!/usr/bin/env python3
"""
ajb_streaming_algorithms.py — Streaming/sketching algorithms for AJB diagnostics

Contributed by Claude #10 (M986-M988) for benchmark data stream analysis.

Contains:
  1. ReservoirSamplingL: Algorithm L (Li 1994) with exponential jumps
  2. CountMinSketch: frequency estimation with point_query + heavy_hitter
  3. HyperLogLog: cardinality estimation with merge support

Each algorithm includes diagnostic print statements.
"""

import hashlib
import math
import random
import struct


class ReservoirSamplingL:
    """Algorithm L (Kim-Hung Li, 1994): Reservoir sampling with exponential jumps.

    Faster than Algorithm R: instead of checking every element, compute
    the next element to include by sampling from a geometric distribution.
    Expected O(k*(1+log(n/k))) time vs O(n) for Algorithm R.
    """

    def __init__(self, k, seed=None):
        self.k = k
        self.rng = random.Random(seed)
        self.reservoir = []
        self.n = 0
        self.w = 1.0
        self.next_skip = 0

    def _compute_w_and_skip(self):
        """Compute next skip distance using exponential jumps."""
        self.w *= math.exp(math.log(self.rng.random()) / self.k)
        skip = int(math.floor(math.log(self.rng.random()) / math.log(1 - self.w)))
        return max(skip, 0)

    def add_stream(self, iterable):
        """Process an entire stream with exponential jump optimization."""
        it = iter(iterable)

        # Fill reservoir with first k elements
        for i, item in enumerate(it):
            self.reservoir.append(item)
            self.n += 1
            if len(self.reservoir) >= self.k:
                break

        if len(self.reservoir) < self.k:
            print(f"[AJB_STATE][ReservoirL] stream exhausted during fill: n={self.n}")
            return self.reservoir

        # Initialize weight and first skip
        self.w = math.exp(math.log(self.rng.random()) / self.k)
        skip = self._compute_w_and_skip()
        print(f"[AJB_STATE][ReservoirL] reservoir filled k={self.k}, initial skip={skip}")

        skipped = 0
        for item in it:
            self.n += 1
            if skipped < skip:
                skipped += 1
                continue

            # Replace random element
            j = self.rng.randint(0, self.k - 1)
            old = self.reservoir[j]
            self.reservoir[j] = item
            print(f"[AJB_TRACE][ReservoirL] n={self.n} replace [{j}]:"
                  f" {old}->{item} (skipped {skip})")

            skip = self._compute_w_and_skip()
            skipped = 0

        print(f"[AJB_STATE][ReservoirL] done: n={self.n}, k={self.k}, w={self.w:.6f}")
        return self.reservoir


class CountMinSketch:
    """Count-Min Sketch for frequency estimation.

    Uses d independent hash functions over a table of width w.
    Point query: min across all d rows.
    Heavy hitter: elements with estimated count > threshold * n.
    """

    def __init__(self, width=256, depth=4, seed=42):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seeds = [seed + i * 7919 for i in range(depth)]
        self.total = 0
        print(f"[AJB_STATE][CMS] created: width={width}, depth={depth}")

    def _hash(self, item, row):
        h = hashlib.md5(f"{self.seeds[row]}:{item}".encode()).digest()
        return struct.unpack('<I', h[:4])[0] % self.w

    def update(self, item, count=1):
        self.total += count
        slots = []
        for row in range(self.d):
            col = self._hash(item, row)
            self.table[row][col] += count
            slots.append((row, col, self.table[row][col]))
        if self.total <= 20 or self.total % 1000 == 0:
            print(f"[AJB_TRACE][CMS] update({item}, +{count}): "
                  + " ".join(f"[{r}][{c}]={v}" for r, c, v in slots))

    def point_query(self, item):
        est = min(self.table[row][self._hash(item, row)] for row in range(self.d))
        return est

    def heavy_hitters(self, threshold=0.01):
        """Find items with estimated count > threshold * total.
        Note: requires tracking candidate items externally.
        """
        cutoff = threshold * self.total
        print(f"[AJB_STATE][CMS] heavy_hitter threshold: "
              f"count > {cutoff:.1f} (total={self.total})")
        return cutoff


class HyperLogLog:
    """HyperLogLog cardinality estimator.

    Uses p bits for register addressing (m=2^p registers).
    Each register stores max(leading zeros + 1) of hashed values.
    Cardinality estimated via harmonic mean of 2^(-register) values.
    """

    def __init__(self, p=10, seed=0):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.seed = seed
        self.count = 0
        # Alpha constant for bias correction
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)
        print(f"[AJB_STATE][HLL] created: p={p}, m={self.m}, alpha={self.alpha:.4f}")

    def _hash(self, item):
        h = hashlib.sha256(f"{self.seed}:{item}".encode()).digest()
        return struct.unpack('<Q', h[:8])[0]

    def _leading_zeros(self, val, bits=64):
        if val == 0:
            return bits
        count = 0
        for i in range(bits - 1, -1, -1):
            if val & (1 << i):
                break
            count += 1
        return count

    def add(self, item):
        h = self._hash(item)
        j = h & (self.m - 1)  # register index (low p bits)
        w = h >> self.p  # remaining bits
        rho = self._leading_zeros(w, 64 - self.p) + 1
        old = self.registers[j]
        if rho > old:
            self.registers[j] = rho
            self.count += 1
            if self.count <= 10 or self.count % 500 == 0:
                print(f"[AJB_TRACE][HLL] add({item}): reg[{j}] "
                      f"{old}->{rho} (hash={h:#018x})")

    def estimate(self):
        """Raw HLL estimate with small/large range corrections."""
        indicator = sum(2.0 ** (-r) for r in self.registers)
        raw = self.alpha * self.m * self.m / indicator

        # Small range correction
        zeros = self.registers.count(0)
        if raw <= 2.5 * self.m and zeros > 0:
            corrected = self.m * math.log(self.m / zeros)
            print(f"[AJB_STATE][HLL] small correction: raw={raw:.1f}"
                  f" -> {corrected:.1f} (zeros={zeros})")
            return corrected

        # Large range correction
        two32 = 2 ** 32
        if raw > two32 / 30:
            corrected = -two32 * math.log(1 - raw / two32)
            print(f"[AJB_STATE][HLL] large correction: raw={raw:.1f}"
                  f" -> {corrected:.1f}")
            return corrected

        print(f"[AJB_STATE][HLL] estimate={raw:.1f} (no correction, zeros={zeros})")
        return raw

    def merge(self, other):
        """Merge another HLL into this one (element-wise max of registers)."""
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different p")
        changed = 0
        for i in range(self.m):
            if other.registers[i] > self.registers[i]:
                self.registers[i] = other.registers[i]
                changed += 1
        print(f"[AJB_STATE][HLL] merged: {changed}/{self.m} registers updated")


def demo():
    print("=" * 64)
    print("  AJB Streaming Algorithms Demo")
    print("=" * 64)

    # ReservoirL
    print("\n--- Reservoir Sampling L ---")
    rs = ReservoirSamplingL(k=5, seed=42)
    sample = rs.add_stream(range(100))
    print(f"  Sample: {sample}")

    # Count-Min Sketch
    print("\n--- Count-Min Sketch ---")
    cms = CountMinSketch(width=64, depth=3)
    for x in [1, 2, 1, 3, 1, 1, 2, 4, 1]:
        cms.update(x)
    for x in [1, 2, 3, 4, 5]:
        print(f"  query({x}) = {cms.point_query(x)}")

    # HyperLogLog
    print("\n--- HyperLogLog ---")
    hll = HyperLogLog(p=8)
    for i in range(1000):
        hll.add(f"item_{i}")
    est = hll.estimate()
    print(f"  Actual=1000, Estimate={est:.1f}, Error={abs(est-1000)/1000*100:.1f}%")


if __name__ == "__main__":
    demo()
