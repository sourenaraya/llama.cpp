ARG UBUNTU_VERSION=24.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.8.1
ARG ROCM_VERSION=7.2.1
ARG GCC_VERSION=14
ARG UBUNTU_CODENAME=noble

# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

ARG NODE_VERSION=24

# GPU targets, override with --build-arg if needed
ARG CUDA_DOCKER_ARCH=120
ARG ROCM_DOCKER_ARCH=gfx1030;gfx1100
# empty means -j$(nproc)
ARG BUILD_JOBS=

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

### Build image
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION
ARG ROCM_VERSION
ARG UBUNTU_CODENAME
ARG CUDA_DOCKER_ARCH
ARG ROCM_DOCKER_ARCH
ARG BUILD_JOBS

# ROCm repo, same as used by the official rocm/dev-ubuntu images
RUN apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake python3 python3-pip git libssl-dev libgomp1 ca-certificates curl gnupg && \
    install -d /etc/apt/keyrings && \
    curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/rocm.list && \
    # amd repo wins over ubuntu's own hipcc/rocm-cmake packages
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 && \
    apt-get update && \
    # rocm-dev is only the toolchain, math libs cmake configs come from the -dev packages
    # hipcub (pulls rocprim) is needed by the CUB argsort/top-k/sum/mean path (PR #26592)
    apt-get install -y --no-install-recommends rocm-dev rocblas-dev hipblas-dev hipcub-dev

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# apt lays ROCm out under /opt/rocm-<ver>, point CMake at it
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" ROCM_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_HIP=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF \
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
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
ARG ROCM_VERSION
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
    &&     curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && apt-get install -y --no-install-recommends rocm-libs \
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
