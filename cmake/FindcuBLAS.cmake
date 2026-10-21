#==========================================================================
#  Copyright (C) Codeplay Software Limited
#  Modified 2026 by MistVVK and the XeStrata contributors: a cross build for Windows gives the CUDA toolkit's
#  headers and Windows' import libraries itself.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  For your convenience, a copy of the License has been included in this
#  repository.
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#=========================================================================

# A cross build for Windows from Linux gives CUDA_TOOLKIT_INCLUDE, CUDA_cublas_LIBRARY, CUDA_CUDART_LIBRARY and
# CUDA_CUDA_LIBRARY (Windows' import libraries, from NVIDIA's redistributable archives): FindCUDA finds the build
# machine's toolkit, whose libraries are Linux's
if(CMAKE_CROSSCOMPILING AND WIN32 AND CUDA_cublas_LIBRARY AND CUDA_TOOLKIT_INCLUDE)
  set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
else()
  find_package(CUDA 10.0 REQUIRED)
endif()
get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
# the OpenCL include file from cuda is opencl 1.1 and it is not compatible with DPC++
# the OpenCL include headers 1.2 onward is required. This is used to bypass NVIDIA OpenCL headers
find_path(OPENCL_INCLUDE_DIR
  NAMES CL/cl.h OpenCL/cl.h
  HINTS
    ${OPENCL_INCLUDE_DIR}
    ${SYCL_BINARY_DIR}/../include/
    ${SYCL_BINARY_DIR}/../../include/
  PATH_SUFFIXES sycl
)
# this is work around to avoid duplication half creation in both cuda and SYCL
add_compile_definitions(CUDA_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)


if(NOT TARGET ONEMATH::cuBLAS::cuBLAS)
  add_library(ONEMATH::cuBLAS::cuBLAS SHARED IMPORTED)
  if(USE_ADD_SYCL_TO_TARGET_INTEGRATION)
    find_package_handle_standard_args(cuBLAS
        REQUIRED_VARS
          CUDA_TOOLKIT_INCLUDE
          CUDA_cublas_LIBRARY
          CUDA_LIBRARIES
          CUDA_CUDART_LIBRARY
	  CUDA_CUDA_LIBRARY
    )
    set_target_properties(ONEMATH::cuBLAS::cuBLAS PROPERTIES
        IMPORTED_LOCATION ${CUDA_cublas_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CUDA_LIBRARIES};${CUDA_CUDART_LIBRARY};${CUDA_CUDA_LIBRARY}"
    )
  else()
    find_package_handle_standard_args(cuBLAS
        REQUIRED_VARS
          CUDA_TOOLKIT_INCLUDE
          CUDA_cublas_LIBRARY
          CUDA_LIBRARIES
          CUDA_CUDA_LIBRARY
          OPENCL_INCLUDE_DIR
    )
    set_target_properties(ONEMATH::cuBLAS::cuBLAS PROPERTIES
        IMPORTED_LOCATION ${CUDA_cublas_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR};${CUDA_TOOLKIT_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}"
    )
  endif()
endif()
