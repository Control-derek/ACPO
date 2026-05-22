#!/usr/bin/env bash
set -euo pipefail
METHOD=ar_lopti REGIME=near_onpolicy bash "$(dirname "$0")/run_math_qwen_7b.sh" "$@"
