g++ testEnumerator.cpp -O2 -g -fno-omit-frame-pointer -o test.exe -lglpk
perf record -g -e cpu-clock -o perf/perf.data ./test.exe
perf script -i perf/perf.data &> perf/perf.unfold
../FlameGraph/stackcollapse-perf.pl perf/perf.unfold &> perf/perf.folded
../FlameGraph/flamegraph.pl perf/perf.folded > perf/perf.svg
wslview perf/perf.svg