ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG GCC_VERSION=14
ARG UBUNTU_CODENAME=noble

# GPU targets, override with --build-arg if needed
ARG CUDA_DOCKER_ARCH=120
ARG ROCM_DOCKER_ARCH=gfx1030;gfx1100
# empty means -j$(nproc)
ARG BUILD_JOBS=
# flip to build without the CUDA backend (faster AMD-only validation)
ARG BUILD_CUDA=ON
# all CPU variants (prod) vs native only (fast local validation)
ARG CPU_ALL_VARIANTS=ON
ARG GGML_NATIVE=OFF
# override the base images, e.g. plain ubuntu for AMD-only builds
ARG BASE_BUILD_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

ARG NODE_VERSION=24

# Target the CUDA build image
FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

### Build image
FROM ${BASE_BUILD_CONTAINER} AS build

ARG GCC_VERSION
ARG UBUNTU_CODENAME
ARG CUDA_DOCKER_ARCH
ARG ROCM_DOCKER_ARCH
ARG BUILD_JOBS
ARG BUILD_CUDA

# ROCm 10 repo (new stable.repo.amd.com layout, replaces repo.radeon.com)
RUN apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake python3 python3-pip git libssl-dev libgomp1 ca-certificates curl gnupg && \
    install -d /etc/apt/keyrings && \
    curl -sL https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor > /etc/apt/keyrings/amdrocm.gpg && \
    APTV=$(echo ${UBUNTU_VERSION} | tr -d .) && \
    printf 'Types: deb\nURIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu%s/\nSuites: stable\nComponents: main\nSigned-By: /etc/apt/keyrings/amdrocm.gpg\nEnabled: yes\n' ${APTV} > /etc/apt/sources.list.d/amdrocm-stable.sources && \
    apt-get update && \
    # per-arch runtime + dev metas cover hipblas/rocblas/hipcub/rocprim/comgr/hsa
    apt-get install -y --no-install-recommends $(for a in $(echo ${ROCM_DOCKER_ARCH} | tr ';' ' '); do echo amdrocm-core10.0-$a amdrocm-core-dev10.0-$a; done) && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# /opt/rocm is a symlink hub into core-10.0 (headers, libs, cmake configs)
RUN HIPCXX=/opt/rocm/llvm/bin/clang HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm \
    cmake -B build -DGGML_NATIVE=${GGML_NATIVE} -DGGML_CUDA=${BUILD_CUDA} -DGGML_HIP=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=${CPU_ALL_VARIANTS} -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} -DAMDGPU_TARGETS="${ROCM_DOCKER_ARCH}" \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j${BUILD_JOBS:-$(nproc)}

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_RUN_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
ARG ROCM_DOCKER_ARCH
ARG UBUNTU_CODENAME
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="llama.cpp" \
      org.opencontainers.image.description="LLM inference in C/C++" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 curl ffmpeg ca-certificates gnupg \
    && install -d /etc/apt/keyrings \
    && curl -sL https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor > /etc/apt/keyrings/amdrocm.gpg \
    && APTV=$(echo ${UBUNTU_VERSION} | tr -d .) \
    && printf 'Types: deb\nURIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu%s/\nSuites: stable\nComponents: main\nSigned-By: /etc/apt/keyrings/amdrocm.gpg\nEnabled: yes\n' ${APTV} > /etc/apt/sources.list.d/amdrocm-stable.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends $(for a in $(echo ${ROCM_DOCKER_ARCH} | tr ';' ' '); do echo amdrocm-core10.0-$a; done) \
    && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig \
    && groupadd -g 109 render \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-wheel \
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

# host is set by the caller (--host / LLAMA_ARG_HOST)

COPY --from=build /app/full/llama /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
