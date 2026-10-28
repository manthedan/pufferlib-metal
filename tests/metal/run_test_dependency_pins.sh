#!/bin/bash
# Host-only offline regression tests for pinned build dependency caches.
set -u

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/puffer-dependency-pins.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

source "$ROOT/tools/build_dependencies.sh"

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

fail() {
    echo "dependency pin test FAIL: $*" >&2
    exit 1
}

assert_file_contains() {
    [ -f "$1" ] || fail "missing file $1"
    grep -q "$2" "$1" || fail "$1 does not contain $2"
}

make_archive() {
    local root=$1 name=$2 payload=$3 archive=$4
    mkdir -p "$root/$name"
    printf '%s\n' "$payload" > "$root/$name/payload.txt"
    tar czf "$archive" -C "$root" "$name"
}

TEST_SHA=
dependency_sha256() {
    [ -n "$TEST_SHA" ] || return 1
    printf '%s\n' "$TEST_SHA"
}

ARCHIVE_ROOT="$TMP/archive-root"
WORK="$TMP/archive-work"
mkdir -p "$ARCHIVE_ROOT" "$WORK"
NAME=raylib-5.5_macos
ARCHIVE="$TMP/$NAME.tar.gz"
make_archive "$ARCHIVE_ROOT" "$NAME" fresh "$ARCHIVE"
TEST_SHA=$(sha256_file "$ARCHIVE")
URL="file://$ARCHIVE"

(
    cd "$WORK" || exit 1
    download "$NAME" "$URL"
) || fail "fresh verified download"
assert_file_contains "$WORK/$NAME/payload.txt" fresh
assert_file_contains "$WORK/$NAME/$DEPENDENCY_PIN_MARKER" "url=$URL"
assert_file_contains "$WORK/$NAME/$DEPENDENCY_PIN_MARKER" "sha256=$TEST_SHA"

REUSE_URL="file://$TMP/missing-reuse.tar.gz"
printf 'url=%s\nsha256=%s\n' "$REUSE_URL" "$TEST_SHA" > "$WORK/$NAME/$DEPENDENCY_PIN_MARKER"
(
    cd "$WORK" || exit 1
    download "$NAME" "$REUSE_URL"
) || fail "valid marker should reuse without network"
assert_file_contains "$WORK/$NAME/payload.txt" fresh

rm -rf "$WORK/$NAME"
mkdir -p "$WORK/$NAME"
printf 'legacy\n' > "$WORK/$NAME/legacy.txt"
(
    cd "$WORK" || exit 1
    download "$NAME" "$URL"
) > "$TMP/legacy.log" 2>&1 || fail "legacy cache replacement"
assert_file_contains "$TMP/legacy.log" "Replacing unverified or stale dependency cache"
[ ! -e "$WORK/$NAME/legacy.txt" ] || fail "legacy cache survived replacement"
assert_file_contains "$WORK/$NAME/payload.txt" fresh

rm -rf "$WORK/$NAME"
BAD_SHA=0000000000000000000000000000000000000000000000000000000000000000
TEST_SHA=$BAD_SHA
if (
    cd "$WORK" || exit 1
    download "$NAME" "$URL"
) > "$TMP/checksum.log" 2>&1; then
    fail "wrong checksum was accepted"
fi
[ ! -e "$WORK/$NAME" ] || fail "wrong checksum left an accepted directory"
assert_file_contains "$TMP/checksum.log" "SHA-256 mismatch"
if find "$WORK" -maxdepth 1 -name ".$NAME.download.*" | grep -q .; then
    fail "wrong checksum left a staging directory"
fi

TEST_SHA=$(sha256_file "$ARCHIVE")
mkdir -p "$WORK/$NAME"
printf 'partial\n' > "$WORK/$NAME/interrupted.txt"
mkdir -p "$WORK/.$NAME.download.interrupted/extracted/$NAME"
printf 'leftover\n' > "$WORK/.$NAME.download.interrupted/extracted/$NAME/payload.txt"
(
    cd "$WORK" || exit 1
    download "$NAME" "$URL"
) || fail "interrupted extraction replacement"
[ ! -e "$WORK/$NAME/interrupted.txt" ] || fail "interrupted target was accepted"
assert_file_contains "$WORK/$NAME/payload.txt" fresh

REMOTE="$TMP/fast-nle-remote"
SEED="$TMP/fast-nle-seed"
git init -q --bare "$REMOTE" || fail "init bare fixture"
git init -q "$SEED" || fail "init fixture checkout"
git -C "$SEED" config user.name 'Dependency Pin Test'
git -C "$SEED" config user.email dependency-pin-test@example.invalid
mkdir -p "$SEED/src"
printf 'build/\n' > "$SEED/.gitignore"
printf 'one\n' > "$SEED/src/version.txt"
git -C "$SEED" add .
git -C "$SEED" commit -qm one || fail "first fixture commit"
COMMIT_ONE=$(git -C "$SEED" rev-parse HEAD)
printf 'two\n' > "$SEED/src/version.txt"
git -C "$SEED" commit -qam two || fail "second fixture commit"
COMMIT_TWO=$(git -C "$SEED" rev-parse HEAD)
git -C "$SEED" remote add origin "$REMOTE"
git -C "$SEED" push -q origin HEAD:main || fail "push fixture"
git --git-dir="$REMOTE" symbolic-ref HEAD refs/heads/main
NLE_REPO="file://$REMOTE"
NLE_COMMIT=$COMMIT_TWO

FRESH_NLE="$TMP/fresh/fast-nle"
fetch_fast_nle "$FRESH_NLE" > "$TMP/nle-fresh.log" 2>&1 || fail "atomic fresh fast-nle clone"
[ "$(git -C "$FRESH_NLE" rev-parse HEAD)" = "$COMMIT_TWO" ] || fail "fresh clone is not pinned"
if find "$TMP/fresh" -maxdepth 1 -name '.fast-nle.clone.*' | grep -q .; then
    fail "fresh clone left a staging directory"
fi

mkdir -p "$FRESH_NLE/build"
printf 'ignored\n' > "$FRESH_NLE/build/output"
SAVED_REPO=$NLE_REPO
NLE_REPO="file://$TMP/missing-fast-nle"
fetch_fast_nle "$FRESH_NLE" || fail "checkout at pin was not reused offline"
# A completed web build leaves build-web/dat/nhdat, untracked and not ignored.
mkdir -p "$FRESH_NLE/build-web/dat"
printf 'nhdat\n' > "$FRESH_NLE/build-web/dat/nhdat"
fetch_fast_nle "$FRESH_NLE" || fail "repeat web build rejected untracked build-web output"
NLE_REPO=$SAVED_REPO

# Concurrent install: another build installs the checkout while this one clones.
RACE_NLE="$TMP/race/fast-nle"
git() {
    command git "$@" || return
    if [ "$1" = clone ] && [ ! -e "$RACE_NLE" ]; then
        command git clone -q "$NLE_REPO" "$RACE_NLE" &&
            command git -C "$RACE_NLE" checkout -q --detach "$NLE_COMMIT"
    fi
}
fetch_fast_nle "$RACE_NLE" > "$TMP/nle-race.log" 2>&1 || { unset -f git; fail "lost clone race was not handled"; }
unset -f git
[ ! -e "$RACE_NLE/fast-nle" ] || fail "lost clone race left a nested checkout"
[ "$(git -C "$RACE_NLE" rev-parse HEAD)" = "$NLE_COMMIT" ] || fail "race winner is not pinned"
if find "$TMP/race" -maxdepth 1 -name '.fast-nle.clone.*' | grep -q .; then
    fail "lost clone race left a staging directory"
fi

CLEAN_NLE="$TMP/clean-other"
git clone -q "$NLE_REPO" "$CLEAN_NLE" || fail "clone clean fixture"
git -C "$CLEAN_NLE" checkout -q --detach "$COMMIT_ONE" || fail "checkout old fixture commit"
mkdir -p "$CLEAN_NLE/build" "$CLEAN_NLE/build-web"
printf 'old\n' > "$CLEAN_NLE/build/libnethack.so"
printf 'old\n' > "$CLEAN_NLE/build-web/libnethack.a"
fetch_fast_nle "$CLEAN_NLE" > "$TMP/nle-update.log" 2>&1 || fail "move clean checkout to pin"
[ "$(git -C "$CLEAN_NLE" rev-parse HEAD)" = "$COMMIT_TWO" ] || fail "clean checkout did not move to pin"
[ ! -e "$CLEAN_NLE/build/libnethack.so" ] || fail "stale native library survived the revision change"
[ ! -e "$CLEAN_NLE/build-web/libnethack.a" ] || fail "stale web library survived the revision change"

MODIFIED_NLE="$TMP/modified"
git clone -q "$NLE_REPO" "$MODIFIED_NLE" || fail "clone modified fixture"
printf 'local work\n' > "$MODIFIED_NLE/src/version.txt"
if fetch_fast_nle "$MODIFIED_NLE" > "$TMP/nle-modified.log" 2>&1; then
    fail "modified checkout was accepted"
fi
assert_file_contains "$MODIFIED_NLE/src/version.txt" "local work"
assert_file_contains "$TMP/nle-modified.log" "will not reset or delete"

INVALID_NLE="$TMP/not-a-checkout"
mkdir -p "$INVALID_NLE/src"
printf 'keep me\n' > "$INVALID_NLE/src/local.txt"
if fetch_fast_nle "$INVALID_NLE" > "$TMP/nle-invalid.log" 2>&1; then
    fail "non-Git checkout was accepted"
fi
assert_file_contains "$INVALID_NLE/src/local.txt" "keep me"
assert_file_contains "$TMP/nle-invalid.log" "Move it aside after preserving any work"

PARTIAL_NLE="$TMP/partial-clone"
git clone -q "$NLE_REPO" "$PARTIAL_NLE" || fail "clone partial fixture"
git -C "$PARTIAL_NLE" config remote.origin.promisor TRUE
git -C "$PARTIAL_NLE" config remote.extra.url "$NLE_REPO"
git -C "$PARTIAL_NLE" config remote.extra.promisor false
if fetch_fast_nle "$PARTIAL_NLE" > "$TMP/nle-partial.log" 2>&1; then
    fail "partial clone was accepted"
fi
assert_file_contains "$PARTIAL_NLE/src/version.txt" two
assert_file_contains "$TMP/nle-partial.log" "partial clone"

echo "dependency pin tests PASS"
