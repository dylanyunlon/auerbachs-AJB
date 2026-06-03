#pragma once
// AJB: merge-sort流配置——计算/传输/回传三路流水线
// upstream硬编码3——这里保持相同值但用constexpr函数提供默认值
// 方便未来根据GPU compute capability调整
constexpr size_t AjbDefaultMergeStreams() { return 3; }
constexpr size_t kNumMergeStreams = AjbDefaultMergeStreams();
