# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0

# Only the CUDA driver API is needed: the kernels arrive as pre-compiled SASS
# and are launched through cuModuleLoadData, so there is no device compilation
# here and no nvcc at wheel-build time.
find_package(CUDAToolkit REQUIRED)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.9.2
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobind)
