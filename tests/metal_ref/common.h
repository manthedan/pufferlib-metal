#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace golden {
inline void check(cudaError_t s) {
    if (s != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(s));
        abort();
    }
}
using Shape = std::vector<int64_t>;
#ifdef PRECISION_FLOAT
constexpr const char *PRECISION_NAME = "PRECISION_FLOAT";
#else
constexpr const char *PRECISION_NAME = "BF16";
#endif
// Manifest dtype of a floating tensor element: float -> f32, __nv_bfloat16 -> bf16.
template <class E> constexpr const char *dtype_of() {
    static_assert(std::is_same_v<E, float> || std::is_same_v<E, __nv_bfloat16>,
                  "golden tensors must be float or __nv_bfloat16");
    return std::is_same_v<E, float> ? "f32" : "bf16";
}
// Allocator entries carry only an element size: 2 bytes = bf16 Prec, 4 bytes = float.
inline const char *dtype_of_size(int elem_size) {
    if (elem_size == 4)
        return "f32";
    if (elem_size == 2)
        return "bf16";
    abort();
}
inline int64_t count(const Shape &s) {
    int64_t n = 1;
    for (auto x : s)
        n *= x;
    return n;
}
inline Shape shape(const int64_t *s) {
    Shape v;
    for (int i = 0; i < 8 && s[i]; i++)
        v.push_back(s[i]);
    return v;
}
inline size_t bytes(const std::string &dtype) {
    if (dtype == "f32" || dtype == "i32")
        return 4;
    if (dtype == "bf16") // raw IEEE bfloat16 bits, little endian
        return 2;
    if (dtype == "u64")
        return 8;
    if (dtype == "u8")
        return 1;
    abort();
}
struct Case {
    std::string dir;
    std::vector<std::string> entries, scalars;
    explicit Case(std::string d) : dir(d) { std::filesystem::create_directories(dir); }
    void scalar(const std::string &k, double x) {
        std::ostringstream o;
        o << std::setprecision(17) << "\"" << k << "\":" << x;
        scalars.push_back(o.str());
    }
    void note(const std::string &k, const std::string &s) {
        scalars.push_back("\"" + k + "\":\"" + s + "\"");
    }
    void host(std::string name, const void *p, Shape s, std::string dtype, std::string role) {
        std::string file = name + ".bin";
        std::ofstream f(dir + "/" + file, std::ios::binary);
        f.write((const char *)p, count(s) * bytes(dtype));
        if (!f)
            abort();
        std::ostringstream o;
        o << "\"" << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
        for (size_t i = 0; i < s.size(); i++) {
            if (i)
                o << ",";
            o << s[i];
        }
        o << "],\"file\":\"" << file << "\",\"role\":\"" << role << "\"}";
        entries.push_back(o.str());
    }
    void device(std::string n, const void *p, Shape s, std::string d = "f32",
                std::string r = "output") {
        check(cudaDeviceSynchronize());
        std::vector<unsigned char> v(count(s) * bytes(d));
        check(cudaMemcpy(v.data(), p, v.size(), cudaMemcpyDeviceToHost));
        host(n, v.data(), s, d, r);
    }
    // Prec and Float: the dtype follows the buffer's element type, so Prec is "bf16" in
    // BF16 builds and "f32" under PRECISION_FLOAT, while Float is always "f32".
    template <class T> void tensor(std::string n, T t, std::string r = "output") {
        using E = std::remove_cv_t<std::remove_pointer_t<decltype(t.data)>>;
        device(n, t.data, shape(t.shape), dtype_of<E>(), r);
    }
    void finish() {
        std::ofstream f(dir + "/manifest.json");
        f << "{\n\"schema_version\":1,\n\"byte_order\":\"little\",\n\"upstream_commit\":"
             "\"ea7f989371270d2948f96728bda8aec5e7634a4c\",\n\"precision\":\""
          << PRECISION_NAME << "\",\n\"tensors\":{\n";
        for (size_t i = 0; i < entries.size(); i++)
            f << (i ? ",\n" : "") << entries[i];
        f << "\n},\n\"scalars\":{\n";
        for (size_t i = 0; i < scalars.size(); i++)
            f << (i ? ",\n" : "") << scalars[i];
        f << "\n}\n}\n";
        if (!f)
            abort();
        printf("wrote %s (%zu tensors)\n", dir.c_str(), entries.size());
    }
};
inline void upload(void *p, const std::string &path, size_t n) {
    std::vector<char> b(n);
    std::ifstream f(path, std::ios::binary);
    f.read(b.data(), n);
    if (!f || f.peek() != EOF) {
        fprintf(stderr, "bad input %s (%zu bytes)\n", path.c_str(), n);
        abort();
    }
    check(cudaMemcpy(p, b.data(), n, cudaMemcpyHostToDevice));
}
inline float pattern(size_t i, int seed = 1, float scale = 1) {
    uint32_t x = (uint32_t)i + 0x9e3779b9u * (uint32_t)seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return ((int)(x & 65535) - 32768) * (scale / 32768.f);
}
} // namespace golden
