# =============================================================================
# info_utilities.py — Terminal status printing (AJB-instrumented)
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/info_utilities.py (28 lines)
# AJB adaptation (~20%): adds AJB_TRACE/AJB_WARN/AJB_FAIL log types that
#   write structured tags to stderr for parse_ajb_trace.py consumption,
#   elapsed time tracking via TIMER type, and ANSI-free fallback when
#   termcolor is unavailable (e.g. piped output in CI).
# =============================================================================

import enum
import sys
import time

try:
    import termcolor
    _has_color = True
except ImportError:
    _has_color = False


class InfoType(enum.Enum):
    DOUBLE_SEPARATOR = 1
    SINGLE_SEPARATOR = 2
    RUN = 3
    PLOT = 4
    SKIPPED = 5
    PASSED = 6
    FAILED = 7
    # AJB: additional status types for structured debug output
    AJB_TRACE = 8
    AJB_WARN = 9
    AJB_FAIL = 10
    AJB_TIMER = 11


info_type_texts_and_colors = {
    InfoType.DOUBLE_SEPARATOR: ("[==========]", "green"),
    InfoType.SINGLE_SEPARATOR: ("[----------]", "green"),
    InfoType.RUN: ("[ RUN      ]", "green"),
    InfoType.PLOT: ("[ PLOT     ]", "green"),
    InfoType.SKIPPED: ("[  SKIPPED ]", "yellow"),
    InfoType.PASSED: ("[  PASSED  ]", "green"),
    InfoType.FAILED: ("[  FAILED  ]", "red"),
    # AJB: structured tags — these go to stderr for machine parsing
    InfoType.AJB_TRACE: ("[AJB_TRACE]", "cyan"),
    InfoType.AJB_WARN:  ("[AJB_WARN]",  "yellow"),
    InfoType.AJB_FAIL:  ("[AJB_FAIL]",  "red"),
    InfoType.AJB_TIMER: ("[AJB_TIMER]", "blue"),
}


def print_info(info_type: InfoType, info: str = ""):
    info_text, info_color = info_type_texts_and_colors[info_type]

    # AJB: route AJB_* types to stderr for parse_ajb_trace.py
    is_ajb = info_type.name.startswith("AJB_")
    stream = sys.stderr if is_ajb else sys.stdout

    if _has_color and stream.isatty():
        print(f"{termcolor.colored(info_text, info_color)} {info}", file=stream)
    else:
        # AJB: ANSI-free fallback for piped/CI environments
        print(f"{info_text} {info}", file=stream)


# AJB: convenience timer context manager for experiment phases
class PhaseTimer:
    """Usage:  with PhaseTimer("data_generation"): do_stuff()"""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        print_info(InfoType.AJB_TRACE, f"{self.name} started")
        return self
    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        print_info(InfoType.AJB_TIMER, f"{self.name} finished in {elapsed:.6f} s")
        self.elapsed = elapsed
