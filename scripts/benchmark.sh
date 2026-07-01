#!/usr/bin/env bash
# build + run a benchmark track, printing the table to stdout.
# usage: scripts/benchmark.sh [all|square|shape|tensor] [N] [reps]
# (invoked by `make run`; can also be called directly.)
set -e
cd "$(dirname "$0")/.."
make -s build
./bin/benchmark "${1:-all}" "${2:-4096}" "${3:-20}"
