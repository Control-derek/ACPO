# Avoid echoing shell initialization commands, which may contain credentials.
set +x
source ~/.bashrc 2>/dev/null || true
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate "${ENV_NAME:-verl}"
bash "$1"
