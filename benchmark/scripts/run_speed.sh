#!/bin/bash
for f in benchmark/ga/speed/bench_*.py; do
    python "$f"
done
