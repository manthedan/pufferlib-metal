#!/bin/bash
# Serial Metal regression suites; Bash 3.2 compatible.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
G=${PUFFER_GOLDEN_DIR:-$ROOT/golden}
mode=run; allow_missing=0; suite=default
usage() {
    echo "Usage: $0 [--list|--check-deps] [--suite default|alternate|mutation] [--allow-missing-fixtures]" >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --list) mode=list ;;
        --check-deps) mode=deps ;;
        --allow-missing-fixtures) allow_missing=1 ;;
        --suite) [ $# -ge 2 ] || usage; suite=$2; shift ;;
        *) usage ;;
    esac
    shift
done
case "$suite" in default|alternate|mutation) ;; *) usage ;; esac
fail=0; failed=0; pass=0; skip=0; total=0
if [ "$mode" != list ]; then
    for tool in xcrun clang++ python3 brew cmake git curl tar make shasum flex bison m4; do
        command -v "$tool" >/dev/null || { echo "MISSING tool: $tool"; fail=1; }
    done
    xcrun -sdk macosx -f metal >/dev/null 2>&1 || { echo 'MISSING toolchain: Metal compiler'; fail=1; }
    brew --prefix libomp >/dev/null 2>&1 || { echo 'MISSING dependency: Homebrew libomp'; fail=1; }
    python3 -c 'import sys, numpy; assert sys.version_info >= (3,9)' >/dev/null 2>&1 || {
        echo 'MISSING dependency: Python >=3.9 with numpy'; fail=1;
    }
    if [ "$mode" = run ]; then
        [ "$fail" = 0 ] || exit 1
        mkdir -p build/metal/gates
        export PUFFER_GATE_RUN_DIR=$(mktemp -d "$ROOT/build/metal/gates/run.XXXXXX")
        export PUFFER_GOLDWIDE_OUT="$PUFFER_GATE_RUN_DIR/goldwide"
        echo "Logs: $PUFFER_GATE_RUN_DIR"
    fi
fi
run() {
    local n=$1 fixtures=$2 status before after
    shift 2
    total=$((total + 1))
    if [ "$mode" = list ]; then
        echo "$n${fixtures:+ (fixtures: $fixtures)}"
        return
    fi
    if [ -n "$fixtures" ]; then
        python3 tests/metal/check_fixtures.py "$G" $fixtures
        status=$?
        if [ "$status" = 1 ]; then
            echo "$n SKIP(missing fixture)"
            skip=$((skip + 1))
            [ "$allow_missing" = 1 ] || fail=1
            return
        elif [ "$status" != 0 ]; then
            echo "$n FAIL (invalid fixture or checker error)"
            failed=$((failed + 1)); fail=1
            return
        fi
    fi
    if [ "$mode" = deps ]; then echo "$n dependencies OK"; return; fi
    before=$(python3 tests/metal/root_output_guard.py "$ROOT/puffer") || {
        echo "$n FAIL (cannot fingerprint root ./puffer)"
        failed=$((failed + 1)); fail=1; return
    }
    status=0
    "$@" > "$PUFFER_GATE_RUN_DIR/$n.log" 2>&1 || status=$?
    after=$(python3 tests/metal/root_output_guard.py "$ROOT/puffer") || status=1
    if [ "$before" != "$after" ]; then
        echo "$n: root ./puffer was created, removed or modified by this gate" >> "$PUFFER_GATE_RUN_DIR/$n.log"
        status=1
    fi
    if [ "$status" = 0 ]; then
        echo "$n PASS"
        pass=$((pass + 1))
    else
        echo "$n FAIL ($status; see $PUFFER_GATE_RUN_DIR/$n.log)"
        failed=$((failed + 1)); fail=1
    fi
}
if [ "$suite" = default ]; then
run root_output_guard "" python3 tests/metal/test_root_output_guard.py
run fixture_validation "" python3 tests/metal/test_fixture_validation.py
run golden_bundle "" python3 tests/metal/test_golden_bundle.py
run dependency_pins "" tests/metal/run_test_dependency_pins.sh
run runtime "" tests/metal/run_test_runtime.sh
run rollout "" tests/metal/run_test_rollout.sh
run shared_env "" tests/metal/run_test_shared_env.sh
run wait_detect "" tests/metal/run_test_wait_detect.sh
run obs_layout "" tests/metal/run_test_obs_layout.sh
run kernels "run_a" tests/metal/run_test_kernels.sh "$G/run_a"
run mingru "run_a" tests/metal/run_test_mingru.sh "$G/run_a"
run gemm "run_a" tests/metal/run_test_gemm.sh "$G/run_a"
run ppo_muon "run_a" tests/metal/run_test_ppo_muon.sh "$G/run_a"
run replay "run_a replay_a" tests/metal/run_train_replay.sh "$G"
run env_golden "envs/run_a" tests/metal/run_test_env_golden.sh "$G/envs"
run bf16 "" tests/metal/run_test_bf16.sh
run rollout_bf16 "" tests/metal/run_test_rollout_bf16.sh
run bf16_golden_a "run_bf16" tests/metal/run_test_bf16_golden.sh "$G/run_bf16"
run bf16_golden_b "run_bf16_b" tests/metal/run_test_bf16_golden.sh "$G/run_bf16_b"
run replay_bf16 "run_bf16 replay_bf16" tests/metal/run_train_replay_bf16.sh "$G"
run mingru_bf16 "" tests/metal/run_test_mingru_bf16.sh
run env_bf16_math "" tests/metal/run_test_env_bf16.sh
run env_golden_bf16_a "envs_bf16/run_a" tests/metal/run_test_env_golden.sh --bf16 "$G/envs_bf16"
run env_golden_bf16_b "envs_bf16/run_b" env PUF_ENV_GOLDEN_RUN=run_b tests/metal/run_test_env_golden.sh --bf16 "$G/envs_bf16"
run h64_e2e "" tests/metal/helpers/h64_e2e.sh
run pack_order "" tests/metal/run_test_pack_order.sh
run overlap_e2e "" tests/metal/helpers/overlap_e2e.sh
run gate "" tests/metal/run_test_gate.sh
run rollout_gate "" tests/metal/run_test_rollout_gate.sh
run wide_gemm "" tests/metal/run_test_wide_gemm.sh
run wide_mingru "" tests/metal/run_test_wide.sh
run wide_muon "" tests/metal/run_test_wide_muon.sh
run wide_mutations "" python3 tests/metal/mutate_wide.py
run wide_e2e "" tests/metal/run_test_wide_e2e.sh
run proj_store "" tests/metal/run_test_proj_store.sh
run sigmoid "" env PUF_SIG_REQUIRE_EXACT=1 tests/metal/run_test_sigmoid.sh
run protein "protein" tests/metal/run_test_protein.sh
run protein_public "" tests/metal/run_test_protein_public.sh
run sweep_cli "" tests/metal/run_test_sweep_cli.sh
run eval_label "" tests/metal/run_test_eval_label.sh
run build_help "" tests/metal/run_test_build_help.sh
run memory_preflight "" tests/metal/run_test_memory_preflight.sh
run warm_start "" tests/metal/run_test_warm_start.sh
run checkpoint_layout "" tests/metal/run_test_checkpoint_layout.sh
run guide "" tests/metal/run_test_guide.sh
run log_history "" tests/metal/run_test_log_history.sh
run goldwide "wide_a wide_b wide_bf16 wide_bf16_b wide_reports" tests/metal/run_test_goldwide.sh "$G"
run large_gemm "" tests/metal/run_test_large_gemm.sh
run large_muon "" tests/metal/run_test_large_muon.sh
run large_linear "" tests/metal/run_test_large_linear.sh
run large_mutations "" python3 tests/metal/mutate_large.py
run large_e2e "" tests/metal/run_test_large_e2e.sh
run robot_arm "" bash tests/metal/run_test_robot_arm.sh
fi
if [ "$suite" = default ] || [ "$suite" = alternate ]; then
run sigmoid_ieee "" env PUF_SIG_REQUIRE_EXACT=0 tests/metal/run_test_sigmoid.sh -DPUF_SIG_IEEE_DIV
run ieee_seed "" tests/metal/run_test_ieee_seed.sh
run strict_launch "" tests/metal/run_test_strict_launch.sh
run optouts_math "" tests/metal/run_test_alternate_paths.sh math
run optouts_schedule "" tests/metal/run_test_alternate_paths.sh schedule
run nethack_variants "" python3 tests/metal/test_nethack_variants.py
fi
if [ "$suite" = mutation ]; then
run sigmoid_mutations "" tests/metal/run_test_mutation_controls.sh sigmoid
run proj_store_mutation "" tests/metal/run_test_mutation_controls.sh proj_store
run goldwide_mutations "" tests/metal/run_test_mutation_controls.sh goldwide
run protein_mutations "protein" python3 tests/metal/mutate_protein.py
fi
if [ "$mode" != list ]; then
    echo "GATES: $total total, $pass PASS, $failed FAIL, $skip SKIP(missing fixture), status=$fail ($mode)"
fi
exit "$fail"
