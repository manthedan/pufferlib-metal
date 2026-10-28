#!/bin/bash
# Metal trainer benchmark: the shared benchmark yardstick.
# Run from the repo root.
#
#   tests/metal/bench.sh                       # build breakout, 3 x 20M-step runs
#   tests/metal/bench.sh -n 5 --steps 30000000 # more / longer runs
#   tests/metal/bench.sh --full                # default 55M steps + eval (prints eval score)
#   tests/metal/bench.sh --no-build --bin ./puffer_x
#   tests/metal/bench.sh --ab ./puffer_a ./puffer_b    # interleaved A/B (a b a b ...)
#   tests/metal/bench.sh -- --base.async=0     # extra ./puffer train args after --
#
# Each run prints its "metal_summary" line (written by pufferl.cu after the
# epoch loop; epoch 0 excluded). The report gives medians per binary:
#   sps          steps / wall over epochs >= 1
#   epoch_ms     wall per epoch
#   train_ms     train_impl GPU timeline per epoch (puf_stamp TE_S..TE_E)
#   rollout_ms   rollout wall per epoch (perf/rollout)
#   dispatches / commits / cpu_waits / barriers per epoch
#   cpu_wait_ms, encode_ms (CPU), {train,rollout}_gpu_ms (summed command
#   buffer GPU spans; queues overlap, so these can sum past epoch_ms)
# With --full the eval score (final env/score in the log ini) is reported too.
# Kernel attribution: PUF_METAL_KPROF=1 (or 2 for per-stream/grid rows)
# ./puffer train ... prints a per-kernel GPU-time table at exit.
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi

ENV_NAME=breakout
N=3
STEPS=20000000
FULL=0
BUILD=1
BINS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case $1 in
        -n) N=$2; shift ;;
        --steps) STEPS=$2; shift ;;
        --full) FULL=1 ;;
        --env) ENV_NAME=$2; shift ;;
        --no-build) BUILD=0 ;;
        --bin) BINS=("$2"); shift ;;
        --ab) BINS=("$2" "$3"); BUILD=0; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done
if [ ${#BINS[@]} -eq 0 ]; then BINS=(./puffer); fi
if [ "$BUILD" = "1" ]; then
    ./build.sh "$ENV_NAME" --metal ${PUF_BENCH_BUILD_PRECISION---float} > /dev/null
fi

ARGS=()
if [ "$FULL" = "1" ]; then
    :
else
    ARGS+=(--train.total-timesteps="$STEPS" --base.eval-episodes=0)
fi
ARGS+=(--base.checkpoint-interval=100000)

OUTDIR=$(mktemp -d "${TMPDIR:-/tmp}/puf_bench.XXXXXX")
LOGDIR="$OUTDIR/logs"
echo "bench: env=$ENV_NAME runs=$N full=$FULL steps=$([ $FULL = 1 ] && echo default || echo $STEPS)" \
    "bins=${BINS[*]} extra=${EXTRA[*]:-} out=$OUTDIR"

for ((i = 0; i < N; i++)); do
    for ((b = 0; b < ${#BINS[@]}; b++)); do
        bin=${BINS[$b]}
        log="$OUTDIR/run_${b}_$i.txt"
        "$bin" train "${ARGS[@]}" --base.log-dir="$LOGDIR/$b.$i" "${EXTRA[@]}" > "$log" 2>&1 \
            || { echo "run failed: $bin (see $log)"; tail -5 "$log"; exit 1; }
        line=$(grep -a '^metal_summary' "$log" | tail -1)
        score=""
        if [ "$FULL" = "1" ]; then
            ini=$(ls -t "$LOGDIR/$b.$i"/*/*.ini 2>/dev/null | head -1)
            score=$(grep '^env/score' "$ini" | sed 's/.*,//; s/.*= //')
            line="$line eval_score=$score"
        fi
        echo "$line" > "$OUTDIR/sum_${b}_$i.txt"
        printf '[%s #%d] %s\n' "$bin" "$i" "$(echo "$line" | awk '{for(i=2;i<=NF;i++) if ($i ~ /^(sps|epoch_ms|train_ms|rollout_ms|eval_score)=/) printf "%s ", $i}')"
    done
done

python3 - "$OUTDIR" "${BINS[@]}" <<'EOF'
import glob, statistics, sys
out, bins = sys.argv[1], sys.argv[2:]
keys = ["sps", "epoch_ms", "train_ms", "rollout_ms", "dispatches", "commits",
        "cpu_waits", "barriers", "cpu_wait_ms", "encode_ms", "train_dispatches",
        "train_encode_ms", "train_wait_ms", "train_gpu_ms", "rollout_dispatches",
        "rollout_encode_ms", "rollout_wait_ms", "rollout_gpu_ms", "rollout_env_ms",
        "rollout_model_ms", "rollout_copy_ms", "eval_score"]
cols = []
for b, name in enumerate(bins):
    vals = {}
    for f in sorted(glob.glob(f"{out}/sum_{b}_*.txt")):
        for tok in open(f).read().split()[1:]:
            k, _, v = tok.partition("=")
            try:
                vals.setdefault(k, []).append(float(v))
            except ValueError:
                pass
    cols.append(vals)
print(f"\n{'median (n=%d)' % len(cols[0].get('sps', [])):<20}" + "".join(f"{b:>18}" for b in bins)
      + ("        delta" if len(bins) == 2 else ""))
for k in keys:
    if not any(k in c for c in cols):
        continue
    row = [statistics.median(c[k]) if k in c else float("nan") for c in cols]
    line = f"{k:<20}" + "".join(f"{v:>18.4g}" for v in row)
    if len(row) == 2 and row[0]:
        line += f"   {100 * (row[1] - row[0]) / row[0]:+7.1f}%"
    print(line)
spread = [f"{b}: sps " + " ".join(f"{v:.3g}" for v in c.get("sps", [])) for b, c in zip(bins, cols)]
print("\n".join(spread))
EOF
