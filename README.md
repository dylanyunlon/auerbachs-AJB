# Auerbachs — Adaptive Hash Join on Mixed-Bandwidth GPU Interconnects

> *"Uns ist ganz kannibalisch wohl, als wie fünfhundert Säuen!"*
> — Auerbachs Keller, Faust I

Data partitions from different bandwidth tiers (NVLink vs PCIe) converge
for adaptive hash joins on A6000 ×2 + H100 ×1 + CPU.

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/multi-gpu-sort-merge-join` | [hpides/multi-gpu-sort-merge-join](https://github.com/hpides/multi-gpu-sort-merge-join) | Multi-GPU sort-merge join |
| `upstream/joinrenum` | [Chen-Py/JoinREnum](https://github.com/Chen-Py/JoinREnum) | Random-order enumeration for joins |
