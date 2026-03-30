cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_ARCHITECTURES all)
project(visacd LANGUAGES CXX C CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
set(OPENVDB_CORE_SHARED OFF)
set(TBB_TEST OFF CACHE BOOL "" FORCE)



# -------------------------------
# FetchContent: pybind11 & eigen
# -------------------------------
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)


# -------------------------------
# Sources (exclude OptiX raygen .cu from static lib!)
# -------------------------------
file(GLOB visacd_CPP "src/*.cpp")
file(GLOB visacd_CUDA "cuda/*.cu")

# Filter out .cu files that should only be PTX (not linked)
list(FILTER visacd_CUDA EXCLUDE REGEX "ray_segments.cu")

add_library(visacd_core STATIC ${visacd_CPP} ${visacd_CUDA})

target_include_directories(visacd_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# -------------------------------
# 3rd-party libraries
# -------------------------------
add_library(quickhull STATIC
    3rd/quickhull/QuickHull.cpp
)
target_include_directories(quickhull PUBLIC
    3rd/quickhull
)


add_subdirectory(3rd/qhull)
target_include_directories(visacd_core PRIVATE qhull/src)



add_library(btConvexHull STATIC
    3rd/btConvexHull/btConvexHullComputer.cpp
    3rd/btConvexHull/btAlignedAllocator.cpp
)
target_include_directories(btConvexHull PUBLIC
    3rd/btConvexHull
)

target_include_directories(visacd_core PRIVATE
    3rd/CDT/CDT/include
)

target_link_libraries(visacd_core PRIVATE
    quickhull
    btConvexHull
    openvdb_static
    Eigen3::Eigen
    qhullcpp
)

include(boost)
include(openvdb)

# -------------------------------
# OptiX Integration
# -------------------------------

# Create interface target for OptiX headers
add_library(optix INTERFACE)
target_include_directories(optix INTERFACE ${OptiX_INSTALL_DIR}/include)

# Link it into our lib
target_link_libraries(visacd_core PRIVATE optix)

# CUDA includes
find_package(CUDA REQUIRED)
target_include_directories(visacd_core PRIVATE ${CUDA_INCLUDE_DIRS})

find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
          ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs
)
target_link_libraries(visacd_core PRIVATE ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} dl)

# -------------------------------
# PTX Compilation (for OptiX programs)
# -------------------------------
set(PTX_DIR ${CMAKE_BINARY_DIR}/ptx)
file(MAKE_DIRECTORY ${PTX_DIR})

# List PTX sources explicitly so you control which ones are compiled
set(PTX_SOURCES
    ${CMAKE_SOURCE_DIR}/ptx/ray_segments.cu
)

if(DEFINED ENV{PTX_COMPUTE})
    set(PTX_SM $ENV{PTX_COMPUTE})
else()
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _GPU_COMPUTE_CAP
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_GPU_COMPUTE_CAP)
        string(REPLACE "." "" PTX_SM ${_GPU_COMPUTE_CAP})
        message(STATUS "Auto-detected GPU compute capability: ${_GPU_COMPUTE_CAP} -> sm_${PTX_SM}")
    else()
        set(PTX_SM "75")
        message(STATUS "nvidia-smi not found, defaulting to sm_75")
    endif()
endif()

foreach(ptx_src ${PTX_SOURCES})
    get_filename_component(ptx_file ${ptx_src} NAME_WE)
    set(ptx_out ${PTX_DIR}/${ptx_file}.ptx)
    add_custom_command(
        OUTPUT ${ptx_out}
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_${PTX_SM}
                -I${OptiX_INSTALL_DIR}/include
                ${ptx_src} -o ${ptx_out}
        DEPENDS ${ptx_src}
    )
    list(APPEND PTX_FILES ${ptx_out})
endforeach()

add_custom_target(compile_ptx ALL DEPENDS ${PTX_FILES})
add_dependencies(visacd_core compile_ptx)

# -------------------------------
# pybind11 module
# -------------------------------
pybind11_add_module(visacd bindings.cpp)
target_link_libraries(visacd PRIVATE visacd_core)
install(TARGETS visacd DESTINATION .)
install(FILES ${PTX_DIR}/ray_segments.ptx DESTINATION .)


target_compile_definitions(visacd_core PRIVATE PTX_DIR="${PTX_DIR}")
