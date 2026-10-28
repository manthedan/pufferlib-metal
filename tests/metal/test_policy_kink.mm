// Minimal reproduction: identical FP32 forward outputs, 4x different
// backward gradients when a tiny projection roundoff crosses zero.
#include <cassert>
#include <cmath>
#include "policy_kink_reference.h"
int main() {
    using namespace policy_ref;
    float lo=-0x1p-29f, hi=0x1p-29f;
    auto forward=[](float h) { return .5f * (.5f * (h>=0 ? h+.5f : 1.f/(1.f+expf(-h)))); };
    assert(forward(lo)==forward(hi));
    double a=.25*hidden_slope(lo), b=.25*hidden_slope(hi);
    assert(fabs(b/a-4)<1e-8);
    assert(fabs((a+branch_delta(lo,hi,a))-b)<1e-15);
    assert(fabs((b+branch_delta(hi,lo,b))-a)<1e-15);
    assert(branch_delta(lo,lo,a)==0);
    // A blanket tolerance of 1e-4 cannot compare the two valid derivatives.
    assert(fabs(a-b)/fabs(a)>1e-4);
}
