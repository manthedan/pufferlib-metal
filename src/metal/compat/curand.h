// Metal backend: cuRAND host-API stand-in. Generation runs on the CPU into
// unified memory (after draining the default stream, which is the
// generator's stream). Streams are NOT bit-identical to cuRAND's XORWOW.
#pragma once
#include <cstddef>
#include "cuda_runtime.h"

typedef enum { CURAND_STATUS_SUCCESS = 0 } curandStatus_t;
typedef enum { CURAND_RNG_PSEUDO_DEFAULT = 100 } curandRngType_t;
typedef struct curandGenerator_st* curandGenerator_t;

curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t type);
curandStatus_t curandDestroyGenerator(curandGenerator_t gen);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t gen,
    unsigned long long seed);
curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t n);
curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n,
    float mean, float stddev);
