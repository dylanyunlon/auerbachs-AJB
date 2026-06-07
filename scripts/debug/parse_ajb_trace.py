#!/usr/bin/env python3
"""
parse_ajb_trace.py — AJB trace log parser and analyzer

Parses the structured [AJB_TIMER], [AJB_STATE], [AJB_RESULTS], [AJB_MEM],
[AJB_BP], [AJB_TRACE] output from AJB-instrumented binaries and produces
a diagnostic summary report.

Usage:
  ./ajb_benchmark 2>&1 | python3 parse_ajb_trace.py
  python3 parse_ajb_trace.py trace.log
  python3 parse_ajb_trace.py trace.log --json > summary.json

M951-M960 enhancements:
  - Welford online variance aggregation for same-name timer events
  - Hot-path detection: top-3 cumulative-time timer phases marked [HOT]
  - State transition tracking: detects key=value changes over time
  - Breakpoint trigger statistics: per-label count + mean interval
  - Structural dump formatting: multi-line AJB_STATE → JSON objects
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict, OrderedDict


# ---------------------------------------------------------------------------
# Welford accumulator — incremental mean/variance in a single pass
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Numerically stable online mean/variance via Welford's algorithm,
    extended with Chan's parallel-merge formula for combining partial
    aggregations from different trace segments.

    Unlike collecting all values then computing numpy.std, this uses O(1)
    memory regardless of how many observations arrive, and avoids the
    catastrophic-cancellation issue of the naive (sum-of-squares) formula.

    Extensions over basic Welford:
      - merge(other): O(1) combine via Chan/Knuth parallel formula
      - min/max tracking for range diagnostics
      - coefficient of variation (cv) for jitter detection
    """
    __slots__ = ('n', 'mean', '_m2', '_min', '_max')

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self._min = float('inf')
        self._max = float('-inf')

    def update(self, x):
        """Incorporate one observation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2
        # Track extrema for range diagnostics
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    def merge(self, other):
        """Combine another WelfordAccumulator via Chan's parallel formula.

        This enables hierarchical aggregation: split a trace into segments,
        aggregate each independently, then merge in O(1) per pair. The
        formula preserves numerical stability:
          combined_m2 = m2_a + m2_b + delta^2 * n_a * n_b / (n_a + n_b)
        where delta = mean_b - mean_a.
        """
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean
            self._m2 = other._m2
            self._min = other._min
            self._max = other._max
            return
        combined_n = self.n + other.n
        delta = other.mean - self.mean
        # Chan's parallel formula for combining M2 aggregates
        self._m2 = (self._m2 + other._m2 +
                    delta * delta * self.n * other.n / combined_n)
        # Weighted mean combination
        self.mean = (self.mean * self.n + other.mean * other.n) / combined_n
        self.n = combined_n
        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max

    @property
    def variance(self):
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def stddev(self):
        return math.sqrt(self.variance)

    @property
    def total(self):
        return self.mean * self.n

    @property
    def cv(self):
        """Coefficient of variation — stddev/mean, measures jitter.
        Values >0.5 indicate high variance; >1.0 extreme instability."""
        if self.n < 2 or self.mean == 0:
            return 0.0
        return self.stddev / abs(self.mean)

    def snapshot(self):
        snap = {
            'count': self.n,
            'mean': self.mean,
            'stddev': self.stddev,
            'total': self.total,
            'cv': self.cv,
        }
        if self.n > 0:
            snap['min'] = self._min
            snap['max'] = self._max
            snap['range'] = self._max - self._min
        return snap


# ---------------------------------------------------------------------------
# State transition tracker — tracks key=value changes across time
# ---------------------------------------------------------------------------
class StateTransitionTracker:
    """Extracts key=value pairs from AJB_STATE lines and records each
    value change per key as a transition chain, with numeric delta
    tracking and monotonicity/trend detection.

    The chain for key K is: [(line_no, val, delta), ...]
    Only transitions (val changes) are stored; duplicate consecutive
    values are suppressed. For numeric values, the delta from previous
    observation is computed; for string values, delta is None.

    Trend detection: after all events are ingested, classify each key's
    trajectory as monotonic_up, monotonic_down, oscillating, or stable.
    """

    def __init__(self):
        self._current = {}          # key -> latest value
        self._current_numeric = {}  # key -> latest numeric value (or None)
        self._chains = defaultdict(list)  # key -> [(line_no, value, delta), ...]
        self._kv_re = re.compile(
            r'(\w+)\s*=\s*'             # key=
            r'('
            r'\[[^\]]*\]'               # [bracketed list]
            r'|'
            r'[^\s,;]+(?:\s+\w+)?'      # value token(s)
            r')'
        )

    @staticmethod
    def _try_numeric(val_str):
        """Attempt to coerce a value string to float. Returns (float, True)
        on success, (None, False) on failure."""
        try:
            return float(val_str), True
        except (ValueError, TypeError):
            return None, False

    def ingest(self, line_no, payload):
        """Parse one AJB_STATE payload for key=value pairs.
        Computes numeric deltas when both old and new values are numeric."""
        for m in self._kv_re.finditer(payload):
            key = m.group(1)
            val = m.group(2).strip()
            prev = self._current.get(key)
            if prev is None:
                # First observation — delta is None
                self._chains[key].append((line_no, val, None))
                self._current[key] = val
                num, ok = self._try_numeric(val)
                self._current_numeric[key] = num if ok else None
            elif val != prev:
                # Value changed — compute numeric delta if possible
                num_new, ok_new = self._try_numeric(val)
                num_old = self._current_numeric.get(key)
                if ok_new and num_old is not None:
                    delta = num_new - num_old
                else:
                    delta = None
                self._chains[key].append((line_no, val, delta))
                self._current[key] = val
                self._current_numeric[key] = num_new if ok_new else None

    def _classify_trend(self, chain):
        """Classify the transition pattern for one key.
        Returns one of: 'monotonic_up', 'monotonic_down', 'oscillating',
        'step_change' (single transition), or 'stable' (no transitions).
        """
        deltas = [d for _, _, d in chain if d is not None]
        if len(chain) < 2:
            return 'stable'
        if len(chain) == 2 and len(deltas) <= 1:
            return 'step_change'
        if not deltas:
            # All string transitions — check if values alternate
            vals = [v for _, v, _ in chain]
            unique_vals = len(set(vals))
            if unique_vals == 2 and len(vals) > 2:
                return 'oscillating'
            return 'step_change'
        # Check monotonicity of numeric deltas
        all_pos = all(d > 0 for d in deltas)
        all_neg = all(d < 0 for d in deltas)
        if all_pos:
            return 'monotonic_up'
        if all_neg:
            return 'monotonic_down'
        # Sign changes indicate oscillation
        sign_changes = sum(1 for i in range(1, len(deltas))
                           if (deltas[i] > 0) != (deltas[i-1] > 0))
        if sign_changes > len(deltas) * 0.4:
            return 'oscillating'
        return 'mixed'

    def transitions(self):
        """Return dict of keys that changed at least once, with their chain
        and trend classification."""
        result = {}
        for key, chain in sorted(self._chains.items()):
            if len(chain) >= 2:
                result[key] = chain
        return result

    def trends(self):
        """Return per-key trend classification for keys with 2+ transitions."""
        out = {}
        for key, chain in sorted(self._chains.items()):
            if len(chain) >= 2:
                out[key] = self._classify_trend(chain)
        return out

    def all_keys(self):
        return dict(self._current)


# ---------------------------------------------------------------------------
# Breakpoint statistics — count + interval analysis per BP label
# ---------------------------------------------------------------------------
class BreakpointStats:
    """Tracks [AJB_BP] labels: how many times each fires and the mean
    interval (in line numbers) between consecutive firings.

    Enhancement: uses WelfordAccumulator for interval statistics (mean,
    variance, CV) and classifies each label's firing pattern:
      - 'periodic': CV < 0.3 (consistent spacing)
      - 'bursty':   CV > 1.0 (clustered with long gaps)
      - 'irregular': in between
      - 'single':   only fired once (no interval data)
    """

    def __init__(self):
        self._events = defaultdict(list)  # label -> [line_no, ...]

    def record(self, line_no, label):
        self._events[label].append(line_no)

    def _classify_pattern(self, interval_acc):
        """Classify firing pattern based on coefficient of variation."""
        if interval_acc.n < 2:
            return 'single' if interval_acc.n <= 1 else 'pair'
        cv = interval_acc.cv
        if cv < 0.3:
            return 'periodic'
        elif cv > 1.0:
            return 'bursty'
        else:
            return 'irregular'

    def summary(self):
        out = {}
        for label, positions in sorted(self._events.items()):
            count = len(positions)
            # Use WelfordAccumulator for interval statistics
            interval_acc = WelfordAccumulator()
            if count >= 2:
                for i in range(count - 1):
                    interval_acc.update(positions[i+1] - positions[i])
            pattern = self._classify_pattern(interval_acc)
            entry = {
                'count': count,
                'first_line': positions[0],
                'last_line': positions[-1],
                'pattern': pattern,
            }
            if interval_acc.n > 0:
                entry['avg_interval_lines'] = interval_acc.mean
                entry['interval_stddev'] = interval_acc.stddev
                entry['interval_cv'] = interval_acc.cv
                entry['min_interval'] = interval_acc._min
                entry['max_interval'] = interval_acc._max
            else:
                entry['avg_interval_lines'] = None
            # Print live data flow for debugging
            print(f"[BP_DIAG] {label}: count={count} pattern={pattern}"
                  + (f" interval_cv={interval_acc.cv:.3f}" if interval_acc.n > 0 else ""),
                  file=sys.stderr)
            out[label] = entry
        return out


# ---------------------------------------------------------------------------
# Multi-line state dump → JSON merger
# ---------------------------------------------------------------------------
def merge_state_dumps(state_lines):
    """Merge consecutive AJB_STATE lines that look like a struct dump
    (lines with key=value or indented continuation lines) into single
    JSON objects with recursive nesting support.

    Enhancement over flat merge:
      - Detects indentation-based hierarchy: indented lines become nested
        sub-objects under the parent context
      - Handles array-like values: [1,2,3] → actual JSON arrays
      - Richer type coercion: bool, int, float, percentage strings
      - Groups are separated by context prefix changes or blank gaps
    """
    kv_re = re.compile(r'(\w+)\s*=\s*(.*)')
    groups = []
    current_obj = OrderedDict()
    current_prefix = None
    current_sub = None  # key for currently accumulating sub-object
    sub_obj = OrderedDict()

    def _coerce_value(raw):
        """Attempt to coerce a raw value string to the best-fit type.
        Handles: bool, int, float, percentage, bracketed arrays, strings."""
        v = raw.strip()
        # Boolean
        if v.lower() in ('true', 'yes'):
            return True
        if v.lower() in ('false', 'no'):
            return False
        # None/null
        if v.lower() in ('none', 'null', 'n/a'):
            return None
        # Bracketed array: [1, 2, 3] or [a, b, c]
        if v.startswith('[') and v.endswith(']'):
            inner = v[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',')]
            coerced = []
            for p in parts:
                try:
                    coerced.append(int(p))
                except ValueError:
                    try:
                        coerced.append(float(p))
                    except ValueError:
                        coerced.append(p)
            return coerced
        # Percentage: "42.5%"
        if v.endswith('%'):
            try:
                return float(v[:-1]) / 100.0
            except ValueError:
                pass
        # Integer
        try:
            return int(v)
        except ValueError:
            pass
        # Float
        try:
            return float(v)
        except ValueError:
            pass
        # Multi-word value: take only first token if rest looks like a label
        return v

    def _flush_sub():
        """Flush accumulated sub-object into current_obj."""
        nonlocal current_sub, sub_obj
        if current_sub and sub_obj:
            current_obj[current_sub] = dict(sub_obj)
        current_sub = None
        sub_obj = OrderedDict()

    def _flush_group():
        """Flush current group to output."""
        nonlocal current_obj, current_prefix
        _flush_sub()
        if current_obj:
            groups.append({'context': current_prefix or 'default',
                           'fields': dict(current_obj)})
            current_obj = OrderedDict()

    for raw in state_lines:
        # Detect leading whitespace (indentation for sub-objects)
        stripped = raw.lstrip()
        indent_level = len(raw) - len(stripped)

        # Detect prefix/context label like [AGM] or [Query]
        prefix_match = re.match(r'\[(\w+)\]\s*(.*)', stripped)
        if prefix_match:
            pfx = prefix_match.group(1)
            rest = prefix_match.group(2)
        else:
            pfx = current_prefix or ''
            rest = stripped

        # If prefix changes, flush the accumulated group
        if pfx != current_prefix and current_obj:
            _flush_group()
        current_prefix = pfx

        # Indented lines (indent > 2) are sub-object continuations
        if indent_level > 2 and current_sub:
            sub_kv = kv_re.match(rest)
            if sub_kv:
                sub_obj[sub_kv.group(1)] = _coerce_value(sub_kv.group(2))
            continue

        # If we had a sub-object accumulating, flush it
        if indent_level <= 2:
            _flush_sub()

        # Extract key=value pairs from this line
        pairs = kv_re.findall(rest)
        if pairs:
            for k, v_raw in pairs:
                coerced = _coerce_value(v_raw)
                # If value looks like a sub-object header (ends with ':')
                if isinstance(coerced, str) and coerced.endswith(':'):
                    current_sub = k
                    sub_obj = OrderedDict()
                else:
                    current_obj[k] = coerced

    _flush_group()

    # Print data flow diagnostic
    total_fields = sum(len(g['fields']) for g in groups)
    nested_count = sum(1 for g in groups for v in g['fields'].values()
                       if isinstance(v, dict))
    print(f"[DUMP_DIAG] merged {len(state_lines)} state lines → "
          f"{len(groups)} objects, {total_fields} fields, "
          f"{nested_count} nested sub-objects", file=sys.stderr)

    return groups


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def parse_trace(lines):
    """Parse AJB trace lines into structured sections with diagnostics."""
    timers_raw = []          # (phase, seconds, line_no)
    timer_accums = {}        # phase -> WelfordAccumulator
    states = []              # raw state payloads
    state_lines_with_no = [] # (line_no, payload)
    results = []
    memory = []
    errors = []
    warnings = []
    bp_tracker = BreakpointStats()
    st_tracker = StateTransitionTracker()
    trace_events = []        # raw [AJB_TRACE] payloads

    timer_re = re.compile(r'\[AJB_TIMER\]\s*(?:<<<?\s*)?(.+?)\s*[:=]?\s*([\d.]+)\s*(s|ms|us)')
    mem_re = re.compile(r'\[AJB_MEM\]\s*(\S+).*?(?:RSS|maxRSS)[=:]?\s*(\d+)\s*(kB|KB|MB)')
    state_re = re.compile(r'\[AJB_STATE\]\s*(.*)')
    result_re = re.compile(r'\[AJB_RESULTS?\]\s*(.*)')
    error_re = re.compile(r'\[AJB_ERROR\]\s*(.*)')
    warn_re = re.compile(r'\[AJB_WARN\]\s*(.*)')
    bp_re = re.compile(r'\[AJB_BP\]\s*(.*)')
    trace_re = re.compile(r'\[AJB_TRACE\]\s*(.*)')

    for line_no, line in enumerate(lines, 1):
        line = line.strip()

        # --- TIMER events ---
        m = timer_re.search(line)
        if m:
            phase = m.group(1).strip()
            raw_val = float(m.group(2))
            unit = m.group(3)
            # Normalize to seconds
            if unit == 'ms':
                seconds = raw_val / 1000.0
            elif unit == 'us':
                seconds = raw_val / 1_000_000.0
            else:
                seconds = raw_val

            timers_raw.append({'phase': phase, 'seconds': seconds,
                               'line': line_no})

            # Welford aggregation per phase name
            if phase not in timer_accums:
                timer_accums[phase] = WelfordAccumulator()
            timer_accums[phase].update(seconds)
            continue

        # --- MEM events ---
        m = mem_re.search(line)
        if m:
            label = m.group(1)
            val = int(m.group(2))
            unit = m.group(3)
            rss_kb = val if unit in ('kB', 'KB') else val * 1024
            memory.append({'label': label, 'rss_kb': rss_kb})
            continue

        # --- STATE events ---
        m = state_re.search(line)
        if m:
            payload = m.group(1)
            states.append(payload)
            state_lines_with_no.append((line_no, payload))
            st_tracker.ingest(line_no, payload)
            continue

        # --- RESULT events ---
        m = result_re.search(line)
        if m:
            results.append(m.group(1))
            continue

        # --- BREAKPOINT events ---
        m = bp_re.search(line)
        if m:
            bp_label = m.group(1).strip()
            bp_tracker.record(line_no, bp_label)
            continue

        # --- TRACE events ---
        m = trace_re.search(line)
        if m:
            trace_events.append({'line': line_no, 'payload': m.group(1)})
            continue

        # --- ERROR / WARN ---
        m = error_re.search(line)
        if m:
            errors.append(m.group(1))
        m = warn_re.search(line)
        if m:
            warnings.append(m.group(1))

    # ----- Post-processing -----

    # Welford-aggregated timer statistics with min/max/cv
    timer_stats = {}
    for phase, acc in timer_accums.items():
        timer_stats[phase] = acc.snapshot()

    # Hot-path detection: adaptive threshold based on cumulative %
    # Instead of hardcoded top-3, mark phases that together account
    # for ≥80% of total time (Pareto-like), with a minimum of 1.
    sorted_phases = sorted(timer_stats.items(),
                           key=lambda kv: kv[1]['total'], reverse=True)
    grand_total = sum(s['total'] for s in timer_stats.values())
    hot_phases = set()
    cumul = 0.0
    hot_threshold = 0.80  # Mark phases covering 80% of wall time
    for i, (phase, s) in enumerate(sorted_phases):
        hot_phases.add(phase)
        timer_stats[phase]['hot'] = True
        timer_stats[phase]['hot_rank'] = i + 1
        cumul += s['total']
        # Stop once we've covered the threshold (but always mark at least 1)
        if grand_total > 0 and cumul / grand_total >= hot_threshold and i >= 0:
            break

    # Timer jitter detection: flag phases with CV > 0.5 as unstable
    jittery_phases = []
    for phase, s in timer_stats.items():
        if s.get('cv', 0) > 0.5 and s['count'] >= 3:
            s['jittery'] = True
            jittery_phases.append(phase)

    # Timer outlier detection: individual raw events >2σ from phase mean
    timer_outliers = []
    for event in timers_raw:
        phase = event['phase']
        s = timer_stats.get(phase)
        if s and s['count'] >= 3 and s['stddev'] > 0:
            z_score = abs(event['seconds'] - s['mean']) / s['stddev']
            if z_score > 2.0:
                timer_outliers.append({
                    'phase': phase,
                    'line': event['line'],
                    'value': event['seconds'],
                    'z_score': z_score,
                })

    # State transition chains with delta information
    transitions = st_tracker.transitions()

    # Per-key trend classification
    state_trends = st_tracker.trends()

    # Breakpoint statistics (now with Welford-based interval variance)
    bp_summary = bp_tracker.summary()

    # Struct dump merge (now with nested sub-objects)
    dump_objects = merge_state_dumps(states)

    # Debug: print comprehensive data flow summary to stderr
    print(f"[DIAG] parsed {len(lines)} lines: "
          f"{len(timers_raw)} timer events, "
          f"{len(states)} state events, "
          f"{len(bp_summary)} BP labels, "
          f"{len(trace_events)} trace events", file=sys.stderr)
    if hot_phases:
        pct_covered = (cumul / grand_total * 100) if grand_total > 0 else 0
        print(f"[DIAG] hot paths ({pct_covered:.0f}% of wall time): "
              f"{sorted(hot_phases)}", file=sys.stderr)
    if jittery_phases:
        print(f"[DIAG] jittery phases (CV>0.5): {jittery_phases}",
              file=sys.stderr)
    if timer_outliers:
        print(f"[DIAG] timer outliers (>2σ): {len(timer_outliers)} events",
              file=sys.stderr)
    if transitions:
        print(f"[DIAG] state transitions for keys: "
              f"{sorted(transitions.keys())}", file=sys.stderr)
    if state_trends:
        print(f"[DIAG] state trends: "
              f"{json.dumps(state_trends, indent=None)}", file=sys.stderr)

    return {
        'timers_raw': timers_raw,
        'timer_stats': timer_stats,
        'timer_outliers': timer_outliers,
        'hot_phases': sorted([p for p in hot_phases]),
        'jittery_phases': jittery_phases,
        'memory': memory,
        'states': states,
        'state_transitions': {
            k: [{'line': ln, 'value': v, 'delta': d} for ln, v, d in chain]
            for k, chain in transitions.items()
        },
        'state_trends': state_trends,
        'state_dumps': dump_objects,
        'breakpoints': bp_summary,
        'trace_events': trace_events,
        'results': results,
        'errors': errors,
        'warnings': warnings,
    }


def print_summary(parsed):
    """Print human-readable diagnostic summary."""
    print("=" * 60)
    print(" AJB Trace Diagnostic Summary")
    print("=" * 60)

    # --- Timer breakdown with Welford stats + hot-path + jitter markers ---
    stats = parsed['timer_stats']
    if stats:
        print("\nTiming breakdown (Welford-aggregated):")
        grand_total = sum(s['total'] for s in stats.values())

        for phase, s in sorted(stats.items(),
                                key=lambda kv: kv[1]['total'], reverse=True):
            pct = f" ({100*s['total']/grand_total:.1f}%)" if grand_total > 0 else ""
            hot_tag = " [HOT]" if s.get('hot') else ""
            jitter_tag = " [JITTERY]" if s.get('jittery') else ""
            if s['count'] > 1:
                range_str = ""
                if 'min' in s and 'max' in s:
                    range_str = f"  range=[{s['min']:.6f},{s['max']:.6f}]"
                cv_str = f"  CV={s['cv']:.3f}" if s.get('cv', 0) > 0 else ""
                detail = (f"  n={s['count']}  "
                          f"mean={s['mean']:.6f}s  "
                          f"σ={s['stddev']:.6f}s  "
                          f"cumul={s['total']:.6f}s{pct}"
                          f"{cv_str}{range_str}{hot_tag}{jitter_tag}")
            else:
                detail = f"  {s['total']:.6f}s{pct}{hot_tag}"
            print(f"  {phase:40s}{detail}")
        print(f"  {'GRAND TOTAL':40s}  {grand_total:.6f}s")

    # --- Timer outliers ---
    outliers = parsed.get('timer_outliers', [])
    if outliers:
        print(f"\nTimer outliers ({len(outliers)} events >2σ):")
        for o in outliers:
            print(f"  L{o['line']:>5d}: {o['phase']:30s}  "
                  f"val={o['value']:.6f}s  z={o['z_score']:.2f}")

    # --- Breakpoint trigger statistics ---
    bp = parsed['breakpoints']
    if bp:
        print(f"\nBreakpoint triggers ({len(bp)} labels):")
        for label, info in sorted(bp.items(),
                                   key=lambda kv: kv[1]['count'], reverse=True):
            interval_str = ""
            if info['avg_interval_lines'] is not None:
                interval_str = (f"  avg_interval={info['avg_interval_lines']:.1f}"
                                f"±{info.get('interval_stddev', 0):.1f} lines")
            pattern_tag = f"  [{info.get('pattern', '?')}]"
            print(f"  {label:50s}  ×{info['count']}"
                  f"  lines [{info['first_line']}..{info['last_line']}]"
                  f"{interval_str}{pattern_tag}")

    # --- State transition chains with deltas and trends ---
    transitions = parsed['state_transitions']
    trends = parsed.get('state_trends', {})
    if transitions:
        print(f"\nState transitions ({len(transitions)} keys changed):")
        for key, chain in sorted(transitions.items()):
            trend_tag = f" [{trends.get(key, '?')}]" if key in trends else ""
            parts = []
            for entry in chain:
                v = entry['value']
                d = entry.get('delta')
                if d is not None:
                    parts.append(f"{v}(Δ{d:+g})")
                else:
                    parts.append(v)
            values_str = " → ".join(parts)
            print(f"  {key}: {values_str}{trend_tag}")

    # --- Struct dump objects ---
    dumps = parsed['state_dumps']
    if dumps:
        print(f"\nStructured state dumps ({len(dumps)} objects):")
        for obj in dumps:
            ctx = obj['context']
            fields_json = json.dumps(obj['fields'], indent=None)
            if len(fields_json) > 120:
                fields_json = json.dumps(obj['fields'], indent=2)
            print(f"  [{ctx}] {fields_json}")

    # --- Memory snapshots ---
    if parsed['memory']:
        print("\nMemory snapshots:")
        for m in parsed['memory']:
            print(f"  {m['label']:20s}  RSS={m['rss_kb']} kB "
                  f"({m['rss_kb']/1024:.1f} MB)")

    # --- Trace events ---
    if parsed['trace_events']:
        print(f"\nTrace events ({len(parsed['trace_events'])}):")
        for t in parsed['trace_events']:
            print(f"  L{t['line']:>5d}: {t['payload']}")

    # --- Results ---
    if parsed['results']:
        print("\nResults:")
        for r in parsed['results']:
            print(f"  {r}")

    # --- Warnings ---
    if parsed['warnings']:
        print(f"\nWarnings ({len(parsed['warnings'])}):")
        for w in parsed['warnings']:
            print(f"  ⚠ {w}")

    # --- Errors ---
    if parsed['errors']:
        print(f"\nErrors ({len(parsed['errors'])}):")
        for e in parsed['errors']:
            print(f"  ✗ {e}")

    if not parsed['errors']:
        print("\n✓ No errors detected")


def main():
    parser = argparse.ArgumentParser(description="AJB trace parser (M951-M960)")
    parser.add_argument("file", nargs="?", default="-",
                        help="Trace log file (- for stdin)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON")
    args = parser.parse_args()

    if args.file == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.file) as f:
            lines = f.readlines()

    parsed = parse_trace(lines)

    if args.json:
        json.dump(parsed, sys.stdout, indent=2)
        print()
    else:
        print_summary(parsed)


if __name__ == "__main__":
    main()
