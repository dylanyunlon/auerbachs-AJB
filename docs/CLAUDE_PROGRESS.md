# AJB 项目 Claude 接力开发进度

## 里程碑编号说明
M = Milestone, 每个文件一个M编号

---

## 第一位 Claude 完成: M621-M660
- joinrenum 20个核心头文件的算法级改写（Bucket.hpp, Index.hpp, RangeTree.hpp等）
- GPU部分的初始改写（math_utilities, hybrid_sort, profile_utilities等）
- 建立了算法级改写规范：改数据结构/遍历策略/计算路径，不是注入fprintf

## 第二位 Claude 完成: M661-M719
### M661: radix_sort.cuh 算法级改写 (14%→88%)
- 10项算法改动：guard-continue扁平化、DivideUp、__builtin_clz、ajb_validate_sort等
- 440 insertions, 334 deletions, 157行纯算法

### M701-M715: 15个高噪声文件的算法级改写 (fprintf→algorithm)
- merge_join.cuh: try_accumulate lambda, bitset<64>, drain_stream, move-insert
- merge_sort.cuh: slab allocation, reserve, indexed access, base-case guard
- radix_sort/kernels.cuh: intrinsics, fused scatter, binary search GPU distance
- merge_join/kernels.cuh: overflow-safe midpoint, unsigned long long
- sort_benchmark.cu: adjacent_find, ValidateSettings, ostringstream
- join_benchmark.cu: SaveRelation helper, sort_relation lambda, reserve
- cpu/gpu_{sort,merge}_benchmark.cu: adjacent_find, swap-dealloc
- data_generator.cuh: memset, array seeds, bucket remainder
- options_limits.cuh: IsValidOption/IsInRange generic helpers
- relation_generator.cuh: derive_s_keys_from_r lambda
- key_value_pair.cuh: operator==, struct-write coalescing
- pinned_vector.cuh: null-guard, operator!=
- **Total: 716 insertions, 1327 deletions, fprintf=0**

### M716-M719: 最后4个噪声文件
- config_utilities.cuh: cudaDeviceCanAccessPeer + pair tracking
- error_utilities.cuh: release-mode throw on CUDA error
- parallel_algorithms.cuh: empty-range guard, ZipIt alias
- constants.cuh: static_assert, JoinStreamOverheadBytes
- **Total: 69 insertions, 114 deletions, fprintf=0**

## 第三位 Claude 待完成: M720-M750 (预估)
- [ ] 全项目 compile validation (nvcc dry-run where possible)
- [ ] 删除 src/common/debug_utilities.cuh (纯fprintf文件, 无upstream对应)
- [ ] 清理剩余6个文件中的低密度fprintf (noise<30%, 各5-8行)
- [ ] CMakeLists.txt 适配、build fix

## 第四位 Claude 待完成: M751-M800 (预估)
- [ ] GPU build validation (需要CUDA环境)
- [ ] Runtime测试 + correctness verification
- [ ] 实验执行 + 数据收集

## 第五位 Claude 待完成: M801-M850 (预估)
- [ ] 数据分析 + 论文图表生成
- [ ] 实验结果解读

## 第六位 Claude 待完成: M851-M900 (预估)
- [ ] 论文修订 + camera-ready
- [ ] 可复现性容器 + 提交材料准备

---

## 关键阻塞项
1. **沙箱无GPU** — CUDA代码只能结构验证不能runtime测试
2. **GLPK依赖** — joinrenum测试需要libglpk-dev
3. **ANSI color污染CSV** — upstream termcolor问题未修
