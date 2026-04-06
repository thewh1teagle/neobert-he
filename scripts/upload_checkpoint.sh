#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> [commit-message]"}
MESSAGE=${2:-"add weights"}

uv run hf upload thewh1teagle/bert-char-he "$CHECKPOINT" --include "model.safetensors" --commit-message "$MESSAGE"
