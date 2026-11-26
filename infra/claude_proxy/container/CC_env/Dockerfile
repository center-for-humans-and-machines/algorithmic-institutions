# syntax=docker/dockerfile:1.6
# CPU-only image for the algorithmic-institutions Claude proxy.
#
# The agent container. Torch is the CPU build of 2.5.1 and PyG wheels are
# the CPU wheels for that torch. This is not the Raven stack (torch 1.11
# + CUDA 11.3); that lives in algoinst.def. An RTX 4090 cannot run the
# Raven wheels, so the local GPU image (Dockerfile.gpu) uses the same
# torch 2.5 generation with cu124 wheels.
#
# The algorithmic-institutions checkout is bind-mounted at
# /opt/algorithmic-institutions. Nothing from that repo is copied in.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=900 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/algorithmic-institutions/src \
    WANDB_ENTITY=chm-ml \
    WANDB_PROJECT=algorithmic-institutions

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg openssh-client \
        build-essential pkg-config \
        libgomp1 \
        locales \
        python3 python3-dev python3-venv python3-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        jq ripgrep fd-find tree file xxd unzip zip xz-utils procps git-lfs \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/* /root/.npm

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "uv==0.9.5"

COPY container/CC_env/requirements.local.txt /opt/requirements.local.txt
COPY container/CC_env/constraints.local-cpu.txt /opt/constraints.local-cpu.txt

RUN uv pip install --system \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1" \
    && uv pip install --system \
        --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html \
        -c /opt/constraints.local-cpu.txt \
        torch_scatter torch_sparse torch_cluster torch_spline_conv \
    && uv pip install --system \
        -c /opt/constraints.local-cpu.txt \
        "torch-geometric==2.6.1" \
        -r /opt/requirements.local.txt

RUN python3 - <<'PY'
import torch
import torch_scatter
import torch_geometric
assert not torch.version.cuda
print("torch", torch.__version__, "cpu")
print("torch_geometric", torch_geometric.__version__)
PY

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN set -eux; \
    if ! getent group "${GROUP_ID}" >/dev/null; then \
        groupadd -g "${GROUP_ID}" claude; \
    fi; \
    if ! getent passwd "${USER_ID}" >/dev/null; then \
        useradd -m -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash claude; \
    else \
        existing_user="$(getent passwd "${USER_ID}" | cut -d: -f1)"; \
        usermod -d /home/claude -m "${existing_user}" || true; \
        usermod -l claude "${existing_user}" || true; \
    fi; \
    mkdir -p /home/claude /opt/algorithmic-institutions /opt/algoinst_proxy; \
    chown -R "${USER_ID}:${GROUP_ID}" /home/claude /opt/algorithmic-institutions /opt/algoinst_proxy

# Author of the commits the pipeline makes (compose passes these from .env).
ARG GIT_USER_NAME="Claude Code"
ARG GIT_USER_EMAIL=""
RUN git config --file /home/claude/.gitconfig user.name "${GIT_USER_NAME}" \
    && if [ -n "${GIT_USER_EMAIL}" ]; then \
        git config --file /home/claude/.gitconfig user.email "${GIT_USER_EMAIL}"; \
    fi \
    && chown "${USER_ID}:${GROUP_ID}" /home/claude/.gitconfig

ENV HOME=/home/claude

USER ${USER_ID}:${GROUP_ID}

WORKDIR /opt/algoinst_proxy

CMD ["claude", "--dangerously-skip-permissions"]
