# =============================================================================
# draw.py — AJB-instrumented result visualization
#
# Origin: upstream/joinrenum/draw.py (17 lines, raw plt.plot)
# AJB adaptation (~25%): Freedman-Diaconis bin width for histogram overlay,
#   IQR-based percentile bands (P25/P75 shading), cumulative distribution
#   overlay on secondary axis, [AJB_VIZ] structured tags to stderr for
#   automated figure-quality checks, log-scale Y detection for multi-order
#   data ranges. Robust CSV parsing replaces manual split.
# =============================================================================

import sys
import os
import math
import statistics

try:
    from matplotlib import pyplot as plt
    _has_plt = True
except ImportError:
    _has_plt = False

def ajb_viz_tag(metric, value, ctx=""):
    """Print structured debug tag for downstream parsers."""
    sys.stderr.write(f"[AJB_VIZ] {metric}={value}")
    if ctx:
        sys.stderr.write(f" ctx={ctx}")
    sys.stderr.write("\n")

def freedman_diaconis_bins(data):
    """Bin count via IQR-based Freedman-Diaconis rule — replaces hardcoded bins."""
    if len(data) < 4:
        return max(1, len(data))
    q75, q25 = sorted(data)[int(len(data)*0.75)], sorted(data)[int(len(data)*0.25)]
    iqr = q75 - q25
    if iqr == 0:
        return max(1, int(math.sqrt(len(data))))
    bin_width = 2.0 * iqr / (len(data) ** (1.0 / 3.0))
    span = max(data) - min(data)
    n_bins = max(1, int(math.ceil(span / bin_width)))
    return min(n_bins, 200)  # clamp to avoid degenerate histograms

def read_results(filename):
    """Robust CSV reader — handles 'x, y' and 'x,y' and 'x y' formats."""
    data = []
    with open(filename, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # try comma-separated first, then whitespace
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()
            if len(parts) < 2:
                sys.stderr.write(f"[AJB_VIZ] WARN skip_line={lineno} reason=too_few_fields\n")
                continue
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                sys.stderr.write(f"[AJB_VIZ] WARN skip_line={lineno} reason=parse_error\n")
    return data

def detect_log_scale(values):
    """Auto-detect if Y should be log-scale: True when max/min > 100."""
    pos = [v for v in values if v > 0]
    if len(pos) < 2:
        return False
    return max(pos) / min(pos) > 100

def plot_with_diagnostics(filename, xlabel=None, ylabel=None, title=None):
    data = read_results(filename)
    if not data:
        sys.stderr.write(f"[AJB_VIZ] ERROR empty_data file={filename}\n")
        return

    X = [row[0] for row in data]
    Y = [row[-1] for row in data]

    # --- structured diagnostics ---
    ajb_viz_tag("n_points", len(X), filename)
    ajb_viz_tag("x_range", f"{min(X):.4g}..{max(X):.4g}", filename)
    ajb_viz_tag("y_range", f"{min(Y):.6g}..{max(Y):.6g}", filename)
    if len(Y) > 1:
        ajb_viz_tag("y_median", f"{statistics.median(Y):.6g}", filename)
        ajb_viz_tag("y_stdev", f"{statistics.stdev(Y):.6g}", filename)

    if not _has_plt:
        # text-mode fallback: print summary table
        sys.stderr.write("[AJB_VIZ] matplotlib unavailable, text-mode summary:\n")
        for i in range(0, len(X), max(1, len(X)//10)):
            print(f"  X={X[i]:.4g}  Y={Y[i]:.6g}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # main scatter + line
    ax1.plot(X, Y, "o-", markersize=3, linewidth=1.2, color="#2563eb", label="measured")

    # AJB: percentile bands (P25/P75) when enough points
    if len(Y) > 8:
        window = max(3, len(Y) // 10)
        rolling_p25, rolling_p75 = [], []
        for i in range(len(Y)):
            lo = max(0, i - window // 2)
            hi = min(len(Y), i + window // 2 + 1)
            chunk = sorted(Y[lo:hi])
            rolling_p25.append(chunk[int(len(chunk) * 0.25)])
            rolling_p75.append(chunk[int(len(chunk) * 0.75)])
        ax1.fill_between(X, rolling_p25, rolling_p75, alpha=0.15,
                         color="#7c3aed", label="P25–P75 band")

    if detect_log_scale(Y):
        ax1.set_yscale("log")
        ajb_viz_tag("y_scale", "log", filename)

    ax1.set_xlabel(xlabel or "Number of result tuples")
    ax1.set_ylabel(ylabel or "Time (s)")
    ax1.set_title(title or os.path.splitext(os.path.basename(filename))[0])

    # AJB: cumulative distribution on secondary axis
    ax2 = ax1.twinx()
    y_sorted = sorted(Y)
    cdf = [(i + 1) / len(y_sorted) for i in range(len(y_sorted))]
    ax2.plot(y_sorted, cdf, "--", color="#dc2626", alpha=0.5, linewidth=0.8, label="CDF(Y)")
    ax2.set_ylabel("Cumulative fraction", color="#dc2626", alpha=0.6)
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    ax1.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = filename.rsplit(".", 1)[0] + "_ajb.png"
    plt.savefig(out_path, dpi=150)
    ajb_viz_tag("saved", out_path, filename)
    plt.close()

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "res/res_q1_bmitu.txt"
    plot_with_diagnostics(fn)
