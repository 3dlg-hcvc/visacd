#pragma once


#include <iomanip>
#include <iostream>
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>


namespace neural_acd {

inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(code) << std::endl
              << "      file: " << file << std::endl
              << "      line: " << line << std::endl;

    if (abort) {
      exit(code);
    }
  }
}

inline void optixAssert(OptixResult code, const char *file, int line,
                        bool abort = true) {
  if (code != OPTIX_SUCCESS) {
    std::cerr << "Optix error file: " << file << " code: " << code
              << " line: " << line << std::endl;

    if (abort) {
      exit(code);
    }
  }
}

#define CHECK_CUDA(result)                                                     \
  {                                                                            \
    cudaAssert((result), __FILE__, __LINE__);                                  \
  }
#define CHECK_OPTIX(result)                                                    \
  {                                                                            \
    optixAssert((result), __FILE__, __LINE__);                                 \
  }

// ---- SBT record helpers (taken from OptiX SDK style) ----
template <typename T> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

// Utility: CUDA check
#define CUCHK(stmt)                                                            \
  do {                                                                         \
    cudaError_t e = (stmt);                                                    \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(cudaGetErrorString(e));                         \
  } while (0)
#define OCHK(stmt)                                                         \
  do {                                                                     \
    OptixResult r = (stmt);                                                \
    if (r != OPTIX_SUCCESS) {                                              \
      std::stringstream ss;                                                \
      ss << "OptiX call failed with error code " << r                     \
         << " at " << __FILE__ << ":" << __LINE__;                        \
      throw std::runtime_error(ss.str());                                   \
    }                                                                      \
  } while (0)

inline static void contextLogCallback(unsigned int level, const char *tag,
                               const char *message, void * /*cbdata */
) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << std::endl;
}

inline static OptixDeviceContext createContext() {
  CHECK_CUDA(cudaFree(0)); // initialize CUDA


  // Get the corresponding driver API context
  CUcontext cuContext = nullptr;
  CUresult cuRes = cuCtxGetCurrent(&cuContext);
  if (cuRes != CUDA_SUCCESS || cuContext == nullptr) {
      std::cerr << "Failed to get CUDA context" << std::endl;
      exit(1);
  }

  OptixResult res = optixInit();
  if (res == 7801){
    std::cerr << "OptiX initialization failed: Incompatible driver version. Please ensure that your NVIDIA driver is up to date and supports the required OptiX version." << std::endl;
    exit(EXIT_FAILURE);
  }
  CHECK_OPTIX(res);

  OptixDeviceContextOptions options = {};
  // options.logCallbackFunction = &contextLogCallback;
  options.logCallbackLevel = 4;

  OptixDeviceContext context;
  // CUcontext cuContext = 0; // current context
  CHECK_OPTIX(optixDeviceContextCreate(cuContext, &options, &context));

  return context;
}

} // namespace neural_acd