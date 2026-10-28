// Independent float64 one-layer policy reference, for diagnosing FP32
// preactivations at MinGRU's derivative discontinuity. Test code only.
#pragma once
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace policy_ref {
using D = std::vector<double>;
inline D widen(const float* p, size_t n) { return D(p, p + n); }
inline D mm(const D& a, const D& b, int M, int N, int K, bool ta=false, bool tb=false) {
    D c((size_t)M * N);
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
        M, N, K, 1, a.data(), ta ? M : K, b.data(), tb ? K : N, 0, c.data(), N);
    return c;
}
inline double sig(double x) { return 1 / (1 + exp(-x)); }
inline double hidden_slope(double x) { double s = sig(x); return x >= 0 ? 1 : s * (1-s); }
struct Reference {
    D x, combined, cache, out, grad, gc, gi, gs, dw, dx, ew;
};
inline Reference one_layer(int B, int T, int H, int I, int O, const D& obs,
        const D& we, const D& wd, const D& w, const D& state, const D& term,
        const D& dg, const D& gn) {
    Reference r;
    int R = B*T;
    r.x = mm(obs, we, R, H, I, false, true);
    r.combined = mm(r.x, w, R, 3*H, H, false, true);
    r.grad = mm(dg, wd, R, H, O);
    r.cache.resize((size_t)R*H); r.out.resize((size_t)R*H);
    r.gc.resize((size_t)R*3*H); r.gi.resize((size_t)R*H); r.gs.resize((size_t)B*H);
    for (int b=0; b<B; b++) for (int h=0; h<H; h++) {
        double ht=state[b*H+h];
        for (int t=0; t<T; t++) {
            size_t j=((size_t)b*T+t)*H+h, c=((size_t)b*T+t)*3*H+h;
            if (term[b*T+t] != 0) ht=0;
            r.cache[j]=ht;
            double hid=r.combined[c], z=sig(r.combined[c+H]), p=sig(r.combined[c+2*H]);
            double v=hid>=0 ? hid+.5 : sig(hid);
            ht += (v-ht)*z;
            r.out[j]=p*ht+(1-p)*r.x[j];
        }
        double dh=gn[b*H+h];
        for (int t=T-1; t>=0; t--) {
            size_t j=((size_t)b*T+t)*H+h, c=((size_t)b*T+t)*3*H+h;
            double hid=r.combined[c], z=sig(r.combined[c+H]), p=sig(r.combined[c+2*H]);
            double v=hid>=0 ? hid+.5 : sig(hid), hp=r.cache[j], ht=hp+(v-hp)*z;
            double total=dh+r.grad[j]*p;
            r.gc[c]=total*z*hidden_slope(hid);
            r.gc[c+H]=total*(v-hp)*z*(1-z);
            r.gc[c+2*H]=r.grad[j]*(ht-r.x[j])*p*(1-p);
            r.gi[j]=r.grad[j]*(1-p);
            dh=term[b*T+t] != 0 ? 0 : total*(1-z);
        }
        r.gs[b*H+h]=dh;
    }
    r.dw=mm(r.gc,r.x,3*H,H,R,true);
    r.dx=mm(r.gc,w,R,H,3*H);
    for (size_t j=0;j<r.dx.size();j++) r.dx[j]+=r.gi[j];
    r.ew=mm(r.dx,obs,H,I,R,true);
    return r;
}
// Change the derivative branch only. Values, gates and recurrence stay on
// the independent reference trajectory; all smooth roundoff remains checked.
inline double branch_delta(double base_hidden, double metal_hidden, double base_gc) {
    if ((base_hidden >= 0) == (metal_hidden >= 0)) return 0;
    return base_gc * (hidden_slope(metal_hidden)/hidden_slope(base_hidden)-1);
}
}
