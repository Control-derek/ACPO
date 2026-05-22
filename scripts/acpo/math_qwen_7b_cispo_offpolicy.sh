#!/usr/bin/env bash
set -euo pipefail
METHOD=cispo REGIME=offpolicy bash "$(dirname "$0")/run_math_qwen_7b.sh" "$@"
