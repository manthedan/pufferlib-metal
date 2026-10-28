#!/bin/bash
set -eu
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
source_file=$1
output=$2
shift 2
omp_prefix=$(brew --prefix libomp)
clang++ -std=c++17 -O2 -Wno-narrowing -I"${ROBOT_ARM_SOURCE_ROOT:-.}" -I. -Isrc -Ivendor \
    -Isrc/metal/compat -Iraylib-5.5_macos/include \
    -Xclang -fopenmp -I"$omp_prefix/include" -L"$omp_prefix/lib" -lomp \
    "$@" "$source_file" raylib-5.5_macos/lib/libraylib.a \
    -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
    -o "$output"
