// Metal backend: NVML stand-in. Utilization queries report "not supported"
// and zero; callers only use them for the dashboard.
#pragma once
typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_NOT_SUPPORTED = 3,
} nvmlReturn_t;
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;
static inline nvmlReturn_t nvmlInit(void) { return NVML_SUCCESS; }
static inline nvmlReturn_t nvmlShutdown(void) { return NVML_SUCCESS; }
static inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int, nvmlDevice_t* d) {
    *d = 0;
    return NVML_SUCCESS;
}
static inline nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t, nvmlUtilization_t* u) {
    u->gpu = 0;
    u->memory = 0;
    return NVML_ERROR_NOT_SUPPORTED;
}
