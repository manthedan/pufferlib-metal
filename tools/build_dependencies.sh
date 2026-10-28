# Sourced from the repository root by build.sh and ocean/nethack/web.sh.
RAYLIB_URL=https://github.com/raysan5/raylib/releases/download/5.5
BOX2D_URL=https://github.com/capnspacehook/box2d/releases/download/df25d747
NLE_REPO=https://github.com/FinlaySanders/fast-nle.git
NLE_COMMIT=e7f6568568242745a7f7ebd8f730367123522cac
DEPENDENCY_PIN_MARKER=.pufferlib-dependency-pin

dependency_sha256() {
    case "$1" in
        raylib-5.5_macos) echo 930c67b676963c6cffbd965814664523081ecbf3d30fc9df4211d0064aa6ba39 ;;
        raylib-5.5_linux_amd64) echo 3d95ef03d5b38dfa55c0a16ca122d382134b078f0e5b270b52fe7eae0549c000 ;;
        raylib-5.5_webassembly) echo 798b6bea650e78a60fe49f106a15d92ea4e33efd3aa1b3efa34b0438a14bbf2c ;;
        box2d-macos-arm64) echo a14b0ee5efb262436805c9d0bded5dea3e617d90a9585e8f19236dbef24d6aaf ;;
        box2d-linux-amd64) echo 5d37f7921504c09bf8387638c0096cee2109b01be5deb9e1809ee60693325c30 ;;
        box2d-web) echo 518c31d1c34dcd6f6772a1cf513bf6372df29e4a1e0fb6f32bd55f85bacd4e78 ;;
        *) echo "Error: no pinned checksum for $1" >&2; return 1 ;;
    esac
}

verify_dependency_archive() {
    local archive=$1 expected=$2 actual
    if command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$archive") || return 1
    elif command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$archive") || return 1
    else
        echo 'Error: shasum or sha256sum is required to verify downloads' >&2
        return 1
    fi
    actual=${actual%% *}
    if [ "$actual" != "$expected" ]; then
        echo "Error: SHA-256 mismatch for $archive (expected $expected, got $actual); refusing extraction" >&2
        return 1
    fi
}

dependency_marker_matches() {
    local directory=$1 url=$2 expected=$3 marker
    [ -d "$directory" ] && [ ! -L "$directory" ] || return 1
    [ -f "$directory/$DEPENDENCY_PIN_MARKER" ] || return 1
    marker=$(printf 'url=%s\nsha256=%s' "$url" "$expected")
    [ "$(cat "$directory/$DEPENDENCY_PIN_MARKER")" = "$marker" ]
}

download() {
    local name=$1 url=$2 expected parent base stage archive extracted backup
    expected=$(dependency_sha256 "$name") || return 1
    if dependency_marker_matches "$name" "$url" "$expected"; then
        return 0
    fi

    parent=$(dirname "$name")
    base=$(basename "$name")
    mkdir -p "$parent" || return 1
    if [ -e "$name" ] || [ -L "$name" ]; then
        echo "Replacing unverified or stale dependency cache $name (pin marker missing or mismatched)."
    fi

    stage=$(mktemp -d "$parent/.$base.download.XXXXXX") || return 1
    archive="$stage/archive"
    extracted="$stage/extracted"
    mkdir "$extracted" || { rm -rf "$stage"; return 1; }

    echo "Downloading $name..."
    if ! curl -fLsS "$url" -o "$archive"; then
        rm -rf "$stage"
        return 1
    fi
    if ! verify_dependency_archive "$archive" "$expected"; then
        rm -rf "$stage"
        return 1
    fi
    case "$url" in
        *.zip)
            if ! unzip -q "$archive" -d "$extracted"; then
                rm -rf "$stage"
                return 1
            fi
            ;;
        *)
            if ! tar xf "$archive" -C "$extracted"; then
                rm -rf "$stage"
                return 1
            fi
            ;;
    esac
    if [ ! -d "$extracted/$base" ] || [ -L "$extracted/$base" ]; then
        echo "Error: verified archive for $name did not contain the expected $base directory" >&2
        rm -rf "$stage"
        return 1
    fi
    if ! printf 'url=%s\nsha256=%s\n' "$url" "$expected" > "$extracted/$base/$DEPENDENCY_PIN_MARKER"; then
        rm -rf "$stage"
        return 1
    fi

    backup=
    if [ -e "$name" ] || [ -L "$name" ]; then
        backup=$(mktemp -d "$parent/.$base.previous.XXXXXX") || { rm -rf "$stage"; return 1; }
        if ! mv "$name" "$backup/$base"; then
            rm -rf "$backup" "$stage"
            return 1
        fi
    fi
    if ! mv "$extracted/$base" "$name"; then
        if [ -n "$backup" ]; then
            if [ ! -e "$name" ] && [ ! -L "$name" ] && mv "$backup/$base" "$name"; then
                rm -rf "$backup"
                backup=
            else
                echo "Error: replacement failed; the previous cache remains at $backup/$base" >&2
                backup=
            fi
        fi
        [ -n "$backup" ] && rm -rf "$backup"
        rm -rf "$stage"
        return 1
    fi
    [ -n "$backup" ] && rm -rf "$backup"
    rm -rf "$stage"
}

fast_nle_checkout_valid() {
    local directory=$1 root physical partial
    [ -d "$directory" ] && [ ! -L "$directory" ] || return 1
    git -C "$directory" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
    root=$(git -C "$directory" rev-parse --show-toplevel 2>/dev/null) || return 1
    physical=$(cd "$directory" && pwd -P) || return 1
    root=$(cd "$root" && pwd -P) || return 1
    [ "$root" = "$physical" ] || return 1
    git -C "$directory" rev-parse --verify HEAD >/dev/null 2>&1 || return 1
    [ -d "$directory/src" ] || return 1
    partial=$(git -C "$directory" config --get extensions.partialClone 2>/dev/null || true)
    [ -z "$partial" ] || return 1
    # --bool normalizes TRUE/yes/on/1; any true entry marks a partial clone.
    partial=$(git -C "$directory" config --bool --get-regexp '^remote\..*\.promisor$' 2>/dev/null || true)
    if printf '%s\n' "$partial" | grep -q ' true$'; then return 1; fi
}

fast_nle_require_clean() {
    local directory=$1 status
    # Tracked changes only: untracked files include build outputs (e.g.
    # build-web/dat/nhdat) and a checkout never touches them.
    status=$(git -C "$directory" status --porcelain --untracked-files=no) || return 1
    if [ -n "$status" ]; then
        echo "Error: $directory has local modifications; preserve or remove them before retrying. The build will not reset or delete them." >&2
        return 1
    fi
}

fetch_fast_nle() {
    local directory=$1 head parent base stage checkout
    if [ -e "$directory" ] || [ -L "$directory" ]; then
        if ! fast_nle_checkout_valid "$directory"; then
            echo "Error: $directory is not a complete fast-nle Git checkout (or is a partial clone). Move it aside after preserving any work, then rerun the build." >&2
            return 1
        fi
        fast_nle_require_clean "$directory" || return 1
        head=$(git -C "$directory" rev-parse HEAD) || return 1
        if [ "$head" = "$NLE_COMMIT" ]; then
            return 0
        fi
        echo "Updating clean fast-nle checkout from $head to $NLE_COMMIT ..."
        git -C "$directory" fetch "$NLE_REPO" "$NLE_COMMIT" || return 1
        git -C "$directory" checkout --detach "$NLE_COMMIT" || return 1
        [ "$(git -C "$directory" rev-parse HEAD)" = "$NLE_COMMIT" ] || return 1
        fast_nle_require_clean "$directory" || return 1
        [ -d "$directory/src" ] || return 1
        # Libraries built from the old revision would otherwise be reused.
        echo "Removing fast-nle build outputs from the previous revision ($directory/build, $directory/build-web)."
        rm -rf "$directory/build" "$directory/build-web"
        return 0
    fi

    parent=$(dirname "$directory")
    base=$(basename "$directory")
    mkdir -p "$parent" || return 1
    stage=$(mktemp -d "$parent/.$base.clone.XXXXXX") || return 1
    checkout="$stage/$base"
    echo "Fetching fast-nle at $NLE_COMMIT ..."
    if ! git clone --no-checkout "$NLE_REPO" "$checkout"; then
        rm -rf "$stage"
        return 1
    fi
    if ! git -C "$checkout" checkout --detach "$NLE_COMMIT"; then
        rm -rf "$stage"
        return 1
    fi
    if [ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null)" != "$NLE_COMMIT" ] ||
            ! fast_nle_checkout_valid "$checkout" || ! fast_nle_require_clean "$checkout"; then
        echo "Error: fresh fast-nle clone did not produce a clean checkout at $NLE_COMMIT" >&2
        rm -rf "$stage"
        return 1
    fi
    # Another build may have installed the checkout while this one cloned;
    # mv onto an existing directory would nest the clone inside it.
    if [ -e "$directory" ] || [ -L "$directory" ]; then
        rm -rf "$stage"
        fetch_fast_nle "$directory"
        return
    fi
    if ! mv "$checkout" "$directory"; then
        echo "Error: could not install fresh fast-nle checkout at $directory" >&2
        rm -rf "$stage"
        return 1
    fi
    rm -rf "$stage"
    if [ -d "$directory/$base/.git" ]; then
        # Lost the race after the check above: drop the nested copy.
        rm -rf "${directory:?}/$base"
        fetch_fast_nle "$directory"
        return
    fi
    if [ "$(git -C "$directory" rev-parse HEAD 2>/dev/null)" != "$NLE_COMMIT" ]; then
        echo "Error: installed fast-nle checkout at $directory is not at $NLE_COMMIT" >&2
        return 1
    fi
}
