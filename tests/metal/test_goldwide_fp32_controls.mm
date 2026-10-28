// Exercise the reader's actual exception validator with host-only fixtures.
// Including the reader keeps these controls coupled to its rejection paths.
#define main unused_mingru_main
#include "test_mingru.mm"
#undef main

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 3) return 2;
        std::string mode = argv[1];
        if (mode != "good" && mode != "sign_bound" && mode != "scan" &&
                mode != "dw" && mode != "dx") return 2;
        using namespace policy_ref;
        const int B=1, T=1, H=2, I=2, O=2;
        // Encoder = [.5,.5]; the first hidden dot product cancels exactly.
        D obs={1,1}, we={.25,.25,.25,.25}, wd={.25,.5,.5,.25};
        D w={1,-1,.5,.5,.25,.25,.25,.5,.5,.25,.25,.25};
        D state={.25,.5}, term={0}, dg={.5,.25}, gn={.125,.25};
        Reference r=one_layer(B,T,H,I,O,obs,we,wd,w,state,term,dg,gn);
        auto f=[](const D& a) { return std::vector<float>(a.begin(),a.end()); };
        auto ob=f(obs), ew=f(we), decw=f(wd), wt=f(w), st=f(state), tr=f(term), next=f(gn);
        auto x=f(r.x), comb=f(r.combined), cache=f(r.cache), out=f(r.out), grad=f(r.grad);
        auto gc=f(r.gc), gi=f(r.gi), gs=f(r.gs);
        // Both changes are tiny enough for the projection's 1e-5 comparison.
        // Only the second exceeds gamma_2 * sum(abs(x*w)) = ~1.19e-7.
        comb[0]=mode == "sign_bound" ? -0x1p-22f : -0x1p-30f;
        gc[0]=(float)(r.gc[0]*hidden_slope(comb[0])/hidden_slope(r.combined[0]));
        D actual_gc(gc.begin(),gc.end());
        auto dw=f(mm(actual_gc,r.x,3*H,H,B*T,true));
        D dx64=mm(actual_gc,w,B*T,H,3*H);
        for (size_t j=0;j<dx64.size();j++) dx64[j]+=gi[j];
        auto dx=f(dx64), encdw=f(mm(D(dx.begin(),dx.end()),obs,H,I,B*T,true));
        if (mode == "scan") gc[0]+=1e-3f;
        if (mode == "dw") dw[0]+=1e-3f;
        if (mode == "dx") dx[0]+=1e-3f;
        PrefixScan scan;
        scan.scan_h.data=cache.data(); scan.out.data=out.data();
        scan.grad_combined.data=gc.data(); scan.grad_input.data=gi.data(); scan.grad_state.data=gs.data();
        NSString* dir=[NSString stringWithUTF8String:argv[2]];
        if (![[NSFileManager defaultManager] createDirectoryAtPath:dir
                withIntermediateDirectories:YES attributes:nil error:nil]) return 2;
        NSMutableDictionary* tensors=[NSMutableDictionary dictionary];
        auto save=[&](NSString* name,const D& a) {
            auto data=f(a);
            NSString* file=[name stringByAppendingString:@".bin"];
            if (![[NSData dataWithBytes:data.data() length:data.size()*sizeof(float)]
                    writeToFile:[dir stringByAppendingPathComponent:file] atomically:YES]) exit(2);
            tensors[name]=@{@"file":file};
        };
        save(@"param_2_grad",r.dw); save(@"encoder_grad",r.dx); save(@"param_0_grad",r.ew);
        Case c={dir,tensors,@{}};
        float dl=.5f, dv=.25f;
        std::vector<float> adw, adx, aew;
        bool accepted=policy_kink_reference(c,mode.c_str(),B,T,H,I,O,ob.data(),ew.data(),
            decw.data(),wt.data(),st.data(),tr.data(),&dl,&dv,next.data(),x.data(),comb.data(),
            scan,grad.data(),dw.data(),dx.data(),encdw.data(),adw,adx,aew);
        if (mode == "good" && !accepted) return 2;
        printf("reference control %s: %s\n",mode.c_str(),g_fail ? "REJECTED" : "ACCEPTED");
        return g_fail ? 1 : 0;
    }
}
