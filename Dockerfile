# NVIDIA's Spark fine-tuning playbook uses this ARM64/GB10-capable release.
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${PYTORCH_IMAGE}

# Inherit NVIDIA's compiled CUDA PyTorch, but isolate project Python packages.
RUN python -c "import sys; assert sys.version_info[:2] == (3, 12)" \
    && python -m venv --system-site-packages /opt/chandassu-venv
ENV PATH="/opt/chandassu-venv/bin:${PATH}" \
    PIP_CONSTRAINT="/dev/null" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/jupyter \
    USER=jupyter \
    PYTHONPATH=/workspace/src:/workspace \
    HF_HOME=/home/jupyter/.cache/huggingface

COPY docker/requirements-gx10.txt /opt/requirements-gx10.txt
RUN python -c "import torch; from pathlib import Path; Path('/opt/torch-version.txt').write_text(torch.__version__); Path('/opt/torch-constraint.txt').write_text('torch==' + torch.__version__ + '\n')" \
    && python -m pip install --no-cache-dir -c /opt/torch-constraint.txt -r /opt/requirements-gx10.txt \
    && python -c "import torch; from pathlib import Path; assert torch.__version__ == Path('/opt/torch-version.txt').read_text(); assert torch.version.cuda is not None"

WORKDIR /workspace
# Project code and data are supplied entirely by the runtime bind mount.
COPY --chmod=755 docker/chandassu /usr/local/bin/chandassu
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; import jupyterlab"

# Match the GX10 account so notebooks/checkpoints on the bind mount stay owned by it.
ARG USER_UID=1000
ARG USER_GID=1000
RUN mkdir -p /home/jupyter/.cache/huggingface /home/jupyter/.local/share/jupyter \
    && chown -R ${USER_UID}:${USER_GID} /home/jupyter /workspace
USER ${USER_UID}:${USER_GID}
RUN python -m ipykernel install --user --name chandassu --display-name "Chandassu (GX10 CUDA)"

EXPOSE 8888
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace"]
