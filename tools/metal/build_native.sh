#!/bin/bash
# Metal backend build for the native trainer (macOS / Apple silicon).
# Sourced by ./build.sh for `./build.sh ENV --metal`; relies on its variables
# (ENV, SRC_DIR, OUT, INCLUDES, LINK_ARCHIVES, EXTRA_SRC, EXTRA_CFLAGS, ...).
#
# Shaders are compiled offline (.metal -> .air -> .metallib) and embedded in
# the binary's __PUF_METAL segment: the primary __metallib and lazy __rollgroup
# library. Compile errors surface at build time and ./puffer stays one file.
# A separate segment keeps shader-size edits from moving host globals in BSS.
# Set PUF_METALLIB=path at runtime to load a different metallib.

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: --metal requires macOS" >&2
    exit 1
fi
if [ "$USE_GPU_ENV" = "1" ]; then
    echo "Error: --metal does not support CUDA envs (--cu / $ENV)" >&2
    exit 1
fi
# Default precision is BF16 (as upstream); --float selects FP32. The two
# builds use separate shader/object directories so neither can pick up the
# other's metallib. Custom encoders use the same precision as the learner.
if [ -z "$PRECISION" ]; then
    MTL_PREC_TAG=bf16
else
    MTL_PREC_TAG=f32
fi
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
if ! xcrun -sdk macosx -f metal >/dev/null 2>&1; then
    echo "Error: Metal compiler not found; install Xcode and set DEVELOPER_DIR" >&2
    exit 1
fi

ENV_HEADER="$SRC_DIR/$ENV.h"
if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
    echo "Error: $ENV_HEADER must typedef obs_t"
    exit 1
fi
if [ "$ENV" = "robot_arm" ]; then
    ENV_HEADER="$SRC_DIR/robot_arm_cpu.h"
fi
if [ -n "$OUT" ]; then
    TRAIN_BIN="$OUT"
elif [ -n "$USER_OUTPUT_NAME" ]; then
    TRAIN_BIN="$USER_OUTPUT_NAME"
else
    TRAIN_BIN="puffer"
fi

MTL_BUILD="build/metal/$ENV-$MTL_PREC_TAG"
mkdir -p "$MTL_BUILD"
METALLIB="$MTL_BUILD/puffer.metallib"
bash tools/metal/build_metallib.sh "$METALLIB" $PRECISION
ROLLOUT_METALLIB="${METALLIB%.metallib}.rollout.metallib"

CXX="${CXX:-clang++}"
OMP_INC="-I$OMP_PREFIX/include"
MTL_CXXFLAGS=(
    -std=c++17 "${CLANG_OPT[@]}"
    -Wno-c99-designator -Wno-c99-extensions -Wno-unused-function
    -Wno-unused-variable -Wno-missing-braces -Wno-writable-strings
    -Wno-logical-op-parentheses -Wno-vla-cxx-extension
    -Isrc/metal/compat -Isrc/metal -I. -Isrc -I"$SRC_DIR" -Ivendor
    "${INCLUDES[@]}"
)
OBJS=()

echo "Compiling $ENV_HEADER -> $TRAIN_BIN (Metal)..."
$CXX "${MTL_CXXFLAGS[@]}" -x c++ \
    -Xclang -fopenmp $OMP_INC \
    -DENV_HEADER=\"$ENV_HEADER\" \
    -DENV_NAME=$ENV \
    -DPUFFER_ENV_NAME=\"$ENV\" \
    -DPUFFERLIB_BUILD_MAIN \
    -DPLATFORM_DESKTOP \
    "${EXTRA_CFLAGS[@]}" \
    $PRECISION \
    -c src/metal/pufferl_metal.cpp -o "$MTL_BUILD/pufferl_metal.o"
OBJS+=("$MTL_BUILD/pufferl_metal.o")

for f in runtime gemm launch; do
    $CXX "${MTL_CXXFLAGS[@]}" -fobjc-arc $PRECISION \
        -c "src/metal/$f.mm" -o "$MTL_BUILD/$f.o"
    OBJS+=("$MTL_BUILD/$f.o")
done

for f in $EXTRA_SRC; do
    o="$MTL_BUILD/$(basename "${f%.*}").o"
    ${CC:-clang} "${CLANG_OPT[@]}" -I. -Isrc -I"$SRC_DIR" -Ivendor "${INCLUDES[@]}" \
        -DPLATFORM_DESKTOP "${EXTRA_CFLAGS[@]}" -c "$f" -o "$o"
    OBJS+=("$o")
done

$CXX $LINK_OPT "${OBJS[@]}" \
    "${LINK_ARCHIVES[@]}" \
    "${EXTRA_LDFLAGS[@]}" \
    -framework Metal -framework Foundation -framework Accelerate \
    "${STANDALONE_LDFLAGS[@]}" \
    -L"$OMP_PREFIX/lib" -lomp -lm -lpthread \
    -Wl,-sectcreate,__PUF_METAL,__metallib,"$METALLIB" \
    -Wl,-sectcreate,__PUF_METAL,__rollgroup,"$ROLLOUT_METALLIB" \
    -Wl,-sectcreate,__PUF_METAL,__wide,"${METALLIB%.metallib}.wide.metallib" \
    -o "$TRAIN_BIN"
echo "Built: ./$TRAIN_BIN"
