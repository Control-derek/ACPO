#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-verl}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
VLLM_VERSION=${VLLM_VERSION:-0.8.4}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.52.1}
RAY_VERSION=${RAY_VERSION:-2.49.2}

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if command -v conda >/dev/null 2>&1; then
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
else
  echo "conda is required. Install Miniconda or Anaconda first." >&2
  exit 1
fi

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "${ENV_NAME}"

if ! python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.')[:2])) else 1)" >/dev/null 2>&1; then
  conda deactivate || true
  conda env remove -y -n "${ENV_NAME}" >/dev/null 2>&1 || true
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  conda activate "${ENV_NAME}"
fi

python -m pip install --upgrade pip
python -m pip install packaging ninja wheel
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -e "${ROOT_DIR}"
python -m pip install "vllm==${VLLM_VERSION}" "transformers==${TRANSFORMERS_VERSION}" "ray==${RAY_VERSION}"
python -m pip install datasets click==8.2.1 cachetools==5.5.2 math_verify markdown nltk ipython ipykernel

python -m ipykernel install --user --name "${ENV_NAME}" || true

if [ -f "${ROOT_DIR}/server/math/requirements.txt" ]; then
  python -m pip install -r "${ROOT_DIR}/server/math/requirements.txt"
fi

echo "Setup complete. Activate with: conda activate ${ENV_NAME}"
