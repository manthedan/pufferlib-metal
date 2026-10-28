"""Compile the allocation-free production predicate with architecture metadata mocks."""
from pathlib import Path
import subprocess

source = Path('src/pufferl.cu').read_text()
a = source.index('static int metal_rollout_pack_split(')
b = source.index('\n// Test hook', a)
body = source[a:b]
assert 'if (metal_rollout_pack_split(pufferl)) {' in source
assert 'const int split = metal_rollout_pack_split(p);' in source
header = Path('src/metal/rollout.h').read_text()
a = header.index('inline bool fused_rollout_supported(')
b = header.index('\ninline void fused_rollout(', a)
support = header[a:b]
prefix = r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#define USE_BF16 false
void encoder_forward() {} void mingru_forward() {} void decoder_forward() {} void custom_forward() {}
struct Layer { void (*forward)(); int in_dim, out_dim, hidden_dim, output_dim, num_layers; };
struct Arch { Layer encoder, decoder, network; };
struct Policy { Arch arch; };
struct Vec { int policy_layout[2]; };
struct PuffeRL {
    struct {int total_agents, num_buffers, hidden_size, horizon;} hypers;
    int num_policies; bool is_continuous;
    Vec* vec;
    struct {struct {float* data; long shape[2];} obs, action_mask;} env;
    Policy* policies;
};
int metal_rollout_packed_env() { const char* e=getenv("PUF_METAL_ROLLOUT_PACKED"); return e ? atoi(e) : -1; }
int obs_kind = 0;
int metal_rollout_step_obs(float*) { return obs_kind; }
namespace puf_mtl { SUPPORT }
'''.replace('SUPPORT', support)
main = r'''
int main() {
    Policy pol{}; pol.arch.encoder = {encoder_forward, 16, 64};
    pol.arch.decoder = {decoder_forward, 0, 0, 64, 4};
    pol.arch.network = {mingru_forward, 0, 0, 0, 0, 2};
    Vec vec{{0, 8}};
    PuffeRL p{{64, 8, 64, 16}, 1, false, &vec, {{nullptr,{64,16}}, {nullptr,{64,4}}}, &pol};
    assert(metal_rollout_pack_split(&p) == 4);
    p.num_policies=2; assert(!metal_rollout_pack_split(&p)); p.num_policies=1;
    p.is_continuous=true; assert(!metal_rollout_pack_split(&p)); p.is_continuous=false;
    pol.arch.encoder.forward=custom_forward; assert(!metal_rollout_pack_split(&p)); pol.arch.encoder.forward=encoder_forward;
    pol.arch.network.num_layers=5; assert(!metal_rollout_pack_split(&p)); pol.arch.network.num_layers=2;
    pol.arch.encoder.in_dim=257; assert(!metal_rollout_pack_split(&p)); pol.arch.encoder.in_dim=16;
    pol.arch.decoder.output_dim=32; assert(!metal_rollout_pack_split(&p)); pol.arch.decoder.output_dim=4;
    vec.policy_layout[1]=4; assert(!metal_rollout_pack_split(&p)); vec.policy_layout[1]=8;
    obs_kind=-1; assert(!metal_rollout_pack_split(&p)); obs_kind=0;
    p.env.action_mask.shape[1]=257; assert(!metal_rollout_pack_split(&p)); p.env.action_mask.shape[1]=4;
    for (const char* key : {"PUF_METAL_ROLLOUT_STEP", "PUF_METAL_FUSED_RESET", "PUF_METAL_FUSED_ROLLOUT", "PUF_METAL_ROLLOUT_STEP_H64", "PUF_METAL_ROLLOUT_PACKED"}) {
        setenv(key,"0",1); assert(!metal_rollout_pack_split(&p)); unsetenv(key);
    }
    setenv("PUF_METAL_ROLLOUT_PACKED","3",1); assert(!metal_rollout_pack_split(&p));
    setenv("PUF_METAL_ROLLOUT_PACKED","2",1); assert(metal_rollout_pack_split(&p)==2);
    setenv("PUF_METAL_ROLLOUT_PACKED","4",1);
    p.hypers.hidden_size=32; pol.arch.encoder.out_dim=32; pol.arch.decoder.hidden_dim=32;
    assert(!metal_rollout_pack_split(&p));
    unsetenv("PUF_METAL_ROLLOUT_PACKED"); assert(metal_rollout_pack_split(&p)==2);
}
'''
out = Path('build/metal/pack_eligibility')
out.mkdir(parents=True, exist_ok=True)
for name, code, ok in [
    ('production', body, True),
    ('continuous', body.replace('|| p->is_continuous', '|| false'), False),
    ('multi_policy', body.replace('p->num_policies != 1', 'false'), False),
    ('custom_encoder', body.replace('arch->encoder.forward != encoder_forward', 'false'), False),
    ('restored', body, True),
]:
    file = out / f'{name}.cpp'
    file.write_text('#include <initializer_list>\n' + prefix + code + main)
    binary = out / name
    subprocess.run(['clang++', '-std=c++17', str(file), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary.resolve())], capture_output=True)
    assert (result.returncode == 0) == ok, name
    print('pack_eligibility:', name, 'PASS' if ok else 'mutant CAUGHT')
