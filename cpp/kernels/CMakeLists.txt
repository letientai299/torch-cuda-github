# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0

# The SASS ships as committed .cubin files (LFS); the byte-array header that
# links them into the extension is generated here, on every build, and never
# committed. Compiling the kernels needs a CUDA toolkit and the private sources;
# embedding them needs neither, which is what lets a public build work.
set(CUBIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vector_add/cubins")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(EMBEDDED_CUBINS "${GENERATED_DIR}/embedded_cubins.h")

file(GLOB CUBIN_FILES CONFIGURE_DEPENDS "${CUBIN_DIR}/*.cubin")

# Depends on the .cubin files, so adding or rebuilding one regenerates the
# header; the generator itself rewrites only on change, so an unchanged set does
# not cascade a rebuild of every dependent object.
add_custom_command(
  OUTPUT "${EMBEDDED_CUBINS}"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/embed_cubins.py" --cubin-dir "${CUBIN_DIR}" --output
          "${EMBEDDED_CUBINS}"
  DEPENDS "${CMAKE_SOURCE_DIR}/tools/embed_cubins.py" ${CUBIN_FILES}
  COMMENT "Embedding ${CUBIN_DIR} into ${EMBEDDED_CUBINS}"
  VERBATIM)

add_custom_target(embedded_cubins DEPENDS "${EMBEDDED_CUBINS}")

nanobind_add_module(
  _kernels
  NB_STATIC
  bindings.cpp
  cubin_runtime.cpp
  vector_add/launcher.cpp)

add_dependencies(_kernels embedded_cubins)

target_include_directories(_kernels PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${GENERATED_DIR}")
target_link_libraries(_kernels PRIVATE CUDA::cuda_driver)

install(TARGETS _kernels LIBRARY DESTINATION torch_cuda/libs)
