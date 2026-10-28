"""Mutation-check real assertions using isolated source copies; always keep production intact."""
import os
import pathlib
import shutil
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[2]
golden = pathlib.Path(os.environ.get('PUFFER_GOLDEN_DIR', root / 'golden')).resolve()
check = subprocess.run(['python3', str(root / 'tests/metal/check_fixtures.py'), str(golden), 'protein'])
if check.returncode == 1:
    print('protein_mutations SKIP(missing fixture)')
    raise SystemExit(0)
check.check_returncode()
os.environ['PUF_PROTEIN_SOBOL_TABLE'] = str(golden / 'protein/curand_sobol32.bin')
mutations = [
    ('kernel_offset', 'src/metal/protein_cpu.h', 'dot + off + (1.0 + u)',
     'dot + 2 * off + (1.0 + u)', 'test_protein_numerics.cu'),
    ('gradient_sign', 'src/metal/protein_cpu.h', 'sums[k] += sh[k * 256];',
     'sums[k] -= sh[k * 256];', 'test_protein_numerics.cu'),
    ('prediction_batch', 'src/metal/protein_cpu.h', 'X + start * gp->dim',
     'X', 'test_protein_sweep.cu'),
    ('sobol_offset', 'src/protein.cu', 'protein_sobol(sw->sobol++, out, dim);',
     'protein_sobol(++sw->sobol, out, dim);', 'test_protein_numerics.cu'),
    ('philox_subsequence', 'src/metal/protein_cpu.h', 'states[i].ctr[2] = i;',
     'states[i].ctr[2] = i + 1;', 'metal/test_protein_rng.cpp'),
    ('cost_cap', 'src/protein.cu', 'c < sw->max_suggestion_cost',
     'c > sw->max_suggestion_cost', 'test_protein.cu'),
    ('pareto_prune', 'src/protein.cu', 'if (ng / (nc + PROTEIN_EPSILON) >= 0.5f)',
     'if (ng / (nc + PROTEIN_EPSILON) >= 0.0f)', 'test_protein.cu'),
    ('early_stop', 'src/protein.cu', 'return score < threshold;',
     'return 0;', 'test_protein.cu'),
    ('duplicate_filter', 'src/protein.cu', 'if (p != self && sq <= r2)',
     'if (p != self && sq < 0)', 'test_protein.cu'),
]
out = root / 'build/metal/protein/mutations'
out.mkdir(parents=True, exist_ok=True)
for name, path, before, after, test in mutations:
    with tempfile.TemporaryDirectory(dir=out) as td:
        tree = pathlib.Path(td)
        shutil.copytree(root / 'src/metal/compat', tree / 'src/metal/compat')
        for source in ['src/protein.cu', 'src/metal/protein_cpu.h']:
            shutil.copyfile(root / source, tree / source)
        target = tree / path
        text = target.read_text()
        assert text.count(before) == 1, name
        target.write_text(text.replace(before, after))
        binary = tree / 'test'
        build = subprocess.run(['clang++', '-x', 'c++', '-std=c++17', '-O2', '-w',
                                '-DPUF_METAL', '-I' + str(tree / 'src'),
                                str(root / 'tests' / test), '-framework', 'Accelerate',
                                '-o', str(binary)], capture_output=True, text=True, cwd=root)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(binary)], capture_output=True, text=True, cwd=root)
        (out / (name + '.log')).write_text(run.stdout + run.stderr)
        assert run.returncode != 0, f'SURVIVED {name}'
        print(f'PASS killed {name}: test exit {run.returncode}', flush=True)
# Verify the unmodified baseline after every mutant source copy has been removed.
subprocess.run(['tests/metal/run_test_protein.sh'], check=True, cwd=root)
