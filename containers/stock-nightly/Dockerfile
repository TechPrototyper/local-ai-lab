# Stock vLLM nightly + one cherry-pick, for consumer Blackwell (sm120 RTX 5090 / sm121 DGX Spark GB10).
#   BASE      unmodified vLLM nightly (multi-arch: amd64 for the RTX, arm64 for the GB10)
#   + FlashInfer 0.7.1 (BatchPrefillWithCausalBidirectionalRangesWrapper, FI PR 5189, NVFP4-capable)
#     — skipped automatically once BASE already ships 0.7.1 (vllm#60948)
#   + vllm#46558 @070ce4baf8 (FlashInfer-native mm-prefix = Gemma-4 image spans on NVFP4/FA2), as a patch against BASE
#   + split-KV shim (vLLM general plugin, only active with VLLM_NVFP4_KV_SPLIT_KV=1)
#   + optional GridBook plugin (RTX: --build-arg GRIDBOOK_VERSION=0.8.8; GB10: empty)
# Build: docker build --platform linux/arm64 -t vllm-stock-nightly:gb10 .
#        docker build --platform linux/amd64 --build-arg GRIDBOOK_VERSION=0.8.8 -t vllm-stock-nightly:rtx .
ARG BASE=vllm/vllm-openai:nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe
FROM ${BASE}
ARG FLASHINFER_VERSION=0.7.1
ARG FLASHINFER_CU=cu130
ARG FLASHINFER_INDEX=https://flashinfer.ai/whl/cu130
ARG VLLM_PATCH=pr46558-070ce4baf8-on-7d0b4e57.vllm.patch
ARG GRIDBOOK_VERSION=
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY ${VLLM_PATCH} /opt/cand-c4/vllm.patch
COPY shim/ /opt/cand-c4/shim/
COPY verify.py /opt/cand-c4/verify.py
# FlashInfer swap: remove *all* flashinfer* packages (jit-cache is split into per-arch sm* sub-packages) and
# delete leftover directories — otherwise Python imports flashinfer_jit_cache as a namespace package without
# __version__ and `import flashinfer` dies. flashinfer-cubin 0.7.1 is not on the index; FlashInfer fetches the
# trtllm-gen cubins (mostly sm100) on demand. download-kernels pulls the missing jit-cache sm* sub-packages.
RUN set -eux; \
    FI_HAVE="$(python3 -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')"; \
    if [ "$FI_HAVE" != "$FLASHINFER_VERSION" ]; then \
      python3 -m pip uninstall -y $(python3 -m pip list 2>/dev/null | awk '/^flashinfer/{print $1}'); \
      SP="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
      rm -rf "$SP/flashinfer" "$SP/flashinfer_cubin" "$SP/flashinfer_jit_cache"; \
      python3 -m pip install --no-cache-dir --no-deps --extra-index-url "${FLASHINFER_INDEX}" \
        "flashinfer-python==${FLASHINFER_VERSION}" "flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CU}"; \
      flashinfer download-kernels; \
    fi; \
    VD="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
    patch -p1 --forward --no-backup-if-mismatch -d "$VD" < /opt/cand-c4/vllm.patch; \
    python3 -m pip install --no-cache-dir --no-deps /opt/cand-c4/shim; \
    if [ -n "$GRIDBOOK_VERSION" ]; then python3 -m pip install --no-cache-dir --no-deps "gridbook==${GRIDBOOK_VERSION}"; fi; \
    python3 /opt/cand-c4/verify.py
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
