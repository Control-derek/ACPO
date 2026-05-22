#!/usr/bin/env bash
set -euo pipefail
METHOD=entropy_bottom REGIME=near_onpolicy bash "$(dirname "$0")/run_math_qwen_7b.sh" "$@"
