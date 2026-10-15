#!/usr/bin/env bash
# Compatibility shim. New collection workflow lives in collect_dataset_batch.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED="$SCRIPT_DIR/collect_dataset_batch.sh"
args=()
while (($#)); do
  case "$1" in
    --num-goals) (($# >= 2)) || { echo "ERROR: --num-goals needs a value" >&2; exit 2; }; args+=(--n-goals "$2"); shift 2 ;;
    --worlds) (($# >= 2)) || { echo "ERROR: --worlds needs a value" >&2; exit 2; }; args+=(--scenarios "$2"); shift 2 ;;
    --base-seed) (($# >= 2)) || { echo "ERROR: --base-seed needs a value" >&2; exit 2; }; args+=(--seed-base "$2"); shift 2 ;;
    --stall-timeout|--log-dir) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; args+=("$1" "$2"); shift 2 ;;
    --seeds|--skip-pool-gen|--pool-force|--pool-num-paths|--pool-continue-on-fail)
      echo "ERROR: '$1' belongs to the legacy pool workflow and is not supported. Use --seed-base, generation quotas, and the unified collector." >&2
      exit 2 ;;
    --) args+=("$@"); break ;;
    *) args+=("$1"); shift ;;
  esac
done
echo "NOTICE: collect_dataset_batch_classic_nav.sh is a compatibility wrapper; use collect_dataset_batch.sh directly." >&2
exec "$UNIFIED" "${args[@]}"
