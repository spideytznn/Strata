#!/bin/sh
# Does not initialize CUDA or load a model. Run from the source root.
set -eu
test_root=$(mktemp -d)
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -Itests/host_cuda -Iinclude \
    tests/stage_storage_test.cpp src/core/weights.cpp -o "$test_root/storage"
"$test_root/storage" "${1:?supply native pack directory}"
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -Itests/host_cuda -Iinclude \
    tests/stage_weight_load_test.cpp src/core/weights.cpp -o "$test_root/loader"
"$test_root/loader" "$test_root/fixture"
