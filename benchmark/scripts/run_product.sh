#!/bin/bash
# Usage: ./run_product.sh <product>
PRODUCT="$1"
for f in benchmark/ga/speed/bench_${PRODUCT}.py \
         benchmark/ga/speed/bench_${PRODUCT}_bwd.py \
         benchmark/ga/memory/bench_${PRODUCT}.py \
         benchmark/ga/memory/bench_${PRODUCT}_bwd.py; do
    [[ -f "$f" ]] && python "$f"
done
