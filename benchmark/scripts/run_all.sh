#!/bin/bash
for f in benchmark/ga/speed/bench_*.py benchmark/ga/memory/bench_*.py; do
    python "$f"
done
python benchmark/ga/plot_improvements.py
python benchmark/ga/print_tables.py
