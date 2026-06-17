#!/bin/bash
for f in benchmark/ga/memory/bench_*.py; do
    python "$f"
done
