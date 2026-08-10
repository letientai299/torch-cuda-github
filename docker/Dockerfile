# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0

# Public base image on purpose: a Dockerfile that only builds behind an
# internal registry would pass internal CI and break every external
# contributor, which is one of the failure modes this sandbox exists to catch.
# Swap in a CUDA base to exercise the GPU path.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /workspace/torch-cuda

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

# Kernel build is opt-in; see setup.py. Without it the wheel is pure Python and
# the ops fall back to Triton or eager.
ARG TORCH_CUDA_BUILD_KERNELS=0
ENV TORCH_CUDA_BUILD_KERNELS=${TORCH_CUDA_BUILD_KERNELS}
RUN pip install --no-cache-dir -e .

CMD ["pytest"]
