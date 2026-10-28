"""Compile production host code; kill the pinned upstream binning mutation."""
import pathlib
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[2]
out_root = root / 'build/metal/log_history'
out_root.mkdir(parents=True, exist_ok=True)
out = pathlib.Path(tempfile.mkdtemp(prefix='host-', dir=out_root))
source = (root / 'src/pufferl.cu').read_text()
block = source[source.index('// One dict copy per train log'):source.index('double rollout_start(')]
upstream = (root / 'tests/metal/fixtures/log_history/upstream_log_history_bin_mean.cu.inc').read_text()
start = upstream.index('static void log_history_bin_mean(')
end = upstream.index('\n}\n', start) + 3
old_function = upstream[start:end]
start = block.index('static void log_history_bin_mean(')
end = block.index('\n}\n', start) + 3
mutant = block[:start] + old_function + block[end:]
nan_function = '''static void log_history_bin_mean(PufLogHistory* h, const char* key,
        int points, double* out) {
    (void)h;
    (void)key;
    for (int i = 0; i < points; i++) out[i] = NAN;
}
'''
nan_mutant = block[:start] + nan_function + block[end:]
for name, code in [('clean', block), ('old-binning', mutant),
                   ('nan-binning', nan_mutant)]:
    (out / 'log_history_under_test.h').write_text(code)
    binary = out / name
    subprocess.run(['clang', '-std=c11', '-O1', '-g',
                    '-fsanitize=address,undefined', '-I' + str(root / 'src'),
                    '-I' + str(out), str(root / 'tests/metal/test_log_history.c'),
                    '-o', str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    (out / (name + '.log')).write_text(run.stdout + run.stderr)
    print(run.stdout + run.stderr, end='')
    if name == 'clean':
        assert run.returncode == 0, (name, run.returncode)
    elif name == 'old-binning':
        assert run.returncode == 1 and 'sparse[' in run.stderr, run
        print('PASS mutation: old one-bin-per-sample implementation rejected')
    else:
        assert run.returncode == 1 and 'got nan' in run.stderr.lower(), run
        print('PASS mutation: NaN binning output rejected')
print('Host evidence:', out)
