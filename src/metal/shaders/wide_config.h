#pragma once
// M4 order-preserving tile table. Offline search: tools/metal/tune_wide.py.
// All use 64-row tiles of whole sequences (T divides 64, M is a multiple of 64).
// H    precision   benchmark (M,T)  chunk  threads  projection/dX columns
// 128  FP32        (16384,64)       8      128      128
// 128  BF16        (16384,64)       16     128      128
// 256  FP32        (8192,64)        8      256      256
// 256  BF16        (8192,64)        16     256      256
// These are compile-time choices, shared by the dispatcher and shaders. No
// arithmetic, split count, K traversal or rounding depends on the choice.
#ifndef PUF_WIDE_128_CHUNK
#ifdef PRECISION_FLOAT
#define PUF_WIDE_128_CHUNK 8
#else
#define PUF_WIDE_128_CHUNK 16
#endif
#endif
#ifndef PUF_WIDE_256_CHUNK
#ifdef PRECISION_FLOAT
#define PUF_WIDE_256_CHUNK 8
#else
#define PUF_WIDE_256_CHUNK 16
#endif
#endif
#ifndef PUF_WIDE_128_THREADS
#define PUF_WIDE_128_THREADS 128
#endif
#ifndef PUF_WIDE_256_THREADS
#define PUF_WIDE_256_THREADS 256
#endif
#ifndef PUF_WIDE_128_N
#define PUF_WIDE_128_N 128
#endif
#ifndef PUF_WIDE_256_N
#define PUF_WIDE_256_N 256
#endif
#ifndef PUF_WIDE_TN_M_F32
#define PUF_WIDE_TN_M_F32 64
#endif
#ifndef PUF_WIDE_TN_M_BF16
#define PUF_WIDE_TN_M_BF16 32
#endif
#ifndef PUF_WIDE_TN_N
#define PUF_WIDE_TN_N 64
#endif
#ifndef PUF_WIDE_TN_THREADS
#define PUF_WIDE_TN_THREADS 64
#endif

// H512/H1024 spatial TN tiles. K traversal and split partitions stay fixed.
#ifndef PUF_LARGE_TN_M_F32
#define PUF_LARGE_TN_M_F32 64
#endif
#ifndef PUF_LARGE_TN_M_BF16
#define PUF_LARGE_TN_M_BF16 64
#endif
#ifndef PUF_LARGE_TN_N
#define PUF_LARGE_TN_N 64
#endif
#ifndef PUF_LARGE_TN_THREADS
#define PUF_LARGE_TN_THREADS 64
#endif

// Measured on M4: (1536,512,65536) and (3072,1024,4096), FP32/BF16.
// H1024 uses 64x32/64 threads in both: within 0.3% of the FP32 sweep
// winner, without that candidate's BF16 regression. Ragged encoders use N64.
#ifndef PUF_LARGE_TN_N512_F32
#define PUF_LARGE_TN_N512_F32 128
#endif
#ifndef PUF_LARGE_TN_N512_BF16
#define PUF_LARGE_TN_N512_BF16 64
#endif
#ifndef PUF_LARGE_TN_N1024
#define PUF_LARGE_TN_N1024 32
#endif

#ifndef PUF_LARGE_MUON_GRAM_M
#define PUF_LARGE_MUON_GRAM_M 64
#endif
#ifndef PUF_LARGE_MUON_GRAM_N
#ifdef PRECISION_FLOAT
#define PUF_LARGE_MUON_GRAM_N 64
#else
#define PUF_LARGE_MUON_GRAM_N 32
#endif
#endif
#ifndef PUF_LARGE_MUON_GRAM_THREADS
#define PUF_LARGE_MUON_GRAM_THREADS 64
#endif

// Recurrent NT/dX: 64x64 with four SIMD groups, selected in both precisions.
#ifndef PUF_LARGE_LINEAR_M
#define PUF_LARGE_LINEAR_M 64
#endif
#ifndef PUF_LARGE_LINEAR_N
#define PUF_LARGE_LINEAR_N 64
#endif
#ifndef PUF_LARGE_LINEAR_THREADS
#define PUF_LARGE_LINEAR_THREADS 128
#endif

#ifndef PUF_LARGE_MUON_PRODUCT_M
#ifdef PRECISION_FLOAT
#define PUF_LARGE_MUON_PRODUCT_M 128
#else
#define PUF_LARGE_MUON_PRODUCT_M 64
#endif
#endif
#ifndef PUF_LARGE_MUON_PRODUCT_N
#ifdef PRECISION_FLOAT
#define PUF_LARGE_MUON_PRODUCT_N 64
#else
#define PUF_LARGE_MUON_PRODUCT_N 32
#endif
#endif
#ifndef PUF_LARGE_MUON_PRODUCT_THREADS
#ifdef PRECISION_FLOAT
#define PUF_LARGE_MUON_PRODUCT_THREADS 128
#else
#define PUF_LARGE_MUON_PRODUCT_THREADS 64
#endif
#endif
