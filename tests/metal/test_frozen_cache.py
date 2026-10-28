"""Exercise the actual NetHack cache helper + policy loader with queued CUDA mocks.
No fast-nle/NetHack dependency is required for this weight-derived-state check.
"""
from pathlib import Path
import subprocess

out = Path('build/metal/frozen_cache')
out.mkdir(parents=True, exist_ok=True)
trainer = Path('src/pufferl.cu').read_text()
nh = Path('ocean/nethack/nethack.cu').read_text()

def function(source, signature):
    start = source.index(signature)
    end = source.index('{', start)
    depth = 1
    while depth:
        end += 1
        depth += (source[end] == '{') - (source[end] == '}')
    return source[start:end + 1]

helper = function(nh, 'static void nethack_refresh_weight_cache(')
loader = function(trainer, 'void pufferl_load_policy(')
assert 'nethack_refresh_weight_cache(w, stream);' in function(nh, 'static void nethack_encoder_init_weights(')
restore = next(line for line in nh.splitlines() if 'cudaMemcpyAsync(ew->msg_w.data, ew->msg_w0_dev,' in line)
prefix = r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#define PUFFER_NETHACK
#define NH_MSG_FROZEN
#ifdef TEST_BF16
using precision_t = unsigned short;
#else
using precision_t = float;
#endif
using cudaStream_t = int;
constexpr int NH_MSG_VOCAB = 3, NH_MSG_HID = 4, cudaMemcpyDeviceToDevice = 1;
struct Prec { precision_t* data; };
struct NethackEncoderWeights { Prec msg_w; precision_t* msg_w0_dev = nullptr; };
std::vector<std::function<void()>> pending;
void cudaMalloc(precision_t** p, size_t n) { *p = (precision_t*)malloc(n); }
void cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t stream) {
    assert(stream == 7);
    pending.push_back([=]{memcpy(d, s, n);});
}
void cudaDeviceSynchronize() { for (auto& f : pending) f(); pending.clear(); }
struct Policy { int params_alloc; Prec master_weights, param; struct {void* encoder;} weights; };
struct PuffeRL { int num_policies; Policy* policies; cudaStream_t default_stream; };
void puf_checkpoint_check_layout(int*, const char*) {}
void puf_load_weights_into(Prec, Prec p, cudaStream_t stream, const char* path) {
    assert(stream == 7);
    int base = strcmp(path, "opponent") == 0 ? 200 : 100;
    pending.push_back([=]{for (int i=0;i<12;i++) p.data[i] = base+i;});
}
'''
main = r'''
int main() {
    precision_t table[12];
    NethackEncoderWeights weights{{table}};
    Policy policy{0, {table}, {table}, {&weights}};
    PuffeRL p{1, &policy, 7};
    // Random initialization and the optional NH_MSG_INIT table both differ
    // from the checkpoint. The same helper captures either initial table.
    for (int init : {1, 20}) {
        for (int i=0;i<12;i++) table[i] = init+i;
        nethack_refresh_weight_cache(&weights, 7);
        cudaDeviceSynchronize();
        assert(weights.msg_w0_dev[0] == init);
        pufferl_load_policy(&p, 0, "checkpoint");
        assert(pending.empty());
        for (int i=0;i<12;i++) table[i] = -1;
        auto* ew = &weights;
        cudaStream_t stream = 7;
        RESTORE
        cudaDeviceSynchronize();
        for (int i=0;i<12;i++) assert(table[i] == 100+i);
    }
    // Loading an opponent must not replace the primary policy's cache.
    precision_t other_table[12];
    NethackEncoderWeights other{{other_table}};
    Policy both[2] = {policy, {0, {other_table}, {other_table}, {&other}}};
    p.num_policies = 2; p.policies = both;
    pufferl_load_policy(&p, 1, "opponent");
    assert(other.msg_w0_dev != weights.msg_w0_dev);
    assert(weights.msg_w0_dev[0] == 100 && other.msg_w0_dev[0] == 200);
}
'''.replace('RESTORE', restore)
for name, body, should_pass in [
    ('production', loader, True),
    ('skip_refresh', loader.replace('nethack_refresh_weight_cache(pol->weights.encoder, pufferl->default_stream);', ''), False),
    ('early_refresh', loader.replace('    puf_load_weights_into(', '    nethack_refresh_weight_cache(pol->weights.encoder, pufferl->default_stream);\n    puf_load_weights_into(').replace('    nethack_refresh_weight_cache(pol->weights.encoder, pufferl->default_stream);\n#endif', '#endif'), False),
    ('shared_cache', loader, False),
    ('restored', loader, True),
]:
    src = out / f'{name}.cpp'
    cache = helper
    if name == 'shared_cache':
        cache = cache.replace('if (!ew->msg_w0_dev) cudaMalloc(&ew->msg_w0_dev, bytes);',
                              'static precision_t* shared = nullptr; if (!shared) cudaMalloc(&shared, bytes); ew->msg_w0_dev = shared;')
    src.write_text(prefix + cache + '\n' + body + '\n' + main)
    binary = out / name
    for precision in ['f32', 'bf16']:
        flags = ['-DTEST_BF16'] if precision == 'bf16' else []
        target = binary.with_name(binary.name + '_' + precision)
        subprocess.run(['clang++', '-std=c++17', *flags, str(src), '-o', str(target)], check=True)
        result = subprocess.run([str(target.resolve())], capture_output=True)
        assert (result.returncode == 0) == should_pass, (name, precision)
        print(f'frozen_cache: {precision} {name} ' + ('PASS' if should_pass else 'mutant CAUGHT'))
