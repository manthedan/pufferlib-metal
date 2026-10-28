#define main unused_bf16_golden_main
#include "test_bf16_golden.mm"
#undef main

static bool rounding_regression() {
    // Every finite adjacent pair, both signs: exact values, midpoint ties,
    // and the neighboring doubles on either side (including subnormals).
    for (uint32_t lo=0;lo<0x7f7f;lo++) {
        double a=puf_bf16_bits_to_float(lo), b=puf_bf16_bits_to_float(lo+1);
        double midpoint=(a+b)*.5;
        double inputs[]={a,midpoint,std::nextafter(midpoint,a),std::nextafter(midpoint,b)};
        uint16_t expected[]={(uint16_t)lo,(uint16_t)(lo+(lo&1)),(uint16_t)lo,(uint16_t)(lo+1)};
        for (int j=0;j<4;j++) for (uint16_t sign : {0,0x8000}) {
            double value=sign ? -inputs[j] : inputs[j];
            if (double_to_bf16_rne(value).x != (expected[j]|sign)) {
                fprintf(stderr,"BF16 midpoint regression FAIL: lo=0x%04x sample=%d sign=%04x\n",lo,j,sign);
                return false;
            }
        }
    }
    double overflow=0x1.ffp127;
    if (double_to_bf16_rne(std::nextafter(overflow,0.)).x != 0x7f7f ||
            double_to_bf16_rne(overflow).x != 0x7f80 ||
            double_to_bf16_rne(-overflow).x != 0xff80 ||
            double_to_bf16_rne(INFINITY).x != 0x7f80 ||
            double_to_bf16_rne(-INFINITY).x != 0xff80 ||
            !std::isnan((float)double_to_bf16_rne(NAN))) return false;
    double z=1/(1+exp(-1.875)), ht=1/(1+exp(3.65625));
    double exact=(1-z)*(-.1640625)+z*ht;
    if (double_to_bf16_rne(exact).x != 0x378b ||
            __float2bfloat16((float)exact).x != 0x378a) return false;
    printf("BF16 midpoint/double-rounding regression PASS\n");
    return true;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        std::string mode=argv[1];
        if (mode == "rounding") return rounding_regression() ? 0 : 1;
        if (mode != "good" && mode != "metal_ulp" && mode != "cuda_ulp") return 2;
        bf16 comb[]={bf16(-3.65625f),bf16(1.875f),bf16(0.f)}, state(-.1640625f);
        // Correct f64 reference is 0x378b. The good endpoints are each one
        // ULP away; 0x3789 is two away and must never receive an exception.
        bf16 metal=bf16::from_bits(mode == "metal_ulp" ? 0x3789 : mode == "good" ? 0x378a : 0x378b);
        std::vector<bf16> cuda={bf16::from_bits(mode == "cuda_ulp" ? 0x3789 : mode == "good" ? 0x378c : 0x378b)};
        check_gate_state({nil,nil,nil},mode,1,1,comb,&state,&metal,cuda);
        printf("reference control %s: %s\n",mode.c_str(),g_fail ? "REJECTED" : "ACCEPTED");
        return g_fail ? 1 : 0;
    }
}
