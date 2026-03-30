#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$SCRIPT_DIR/files"
REPO_ROOT="$(pwd)"
BACKUP_DIR="$REPO_ROOT/backup_before_speed_fixes_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Backup dir: $BACKUP_DIR"

for f in \
  train/common/config.py \
  train/common/callbacks.py \
  train/train_sac.py \
  utils/env_utils.py \
  utils/io_utils.py \
  evaluate/evaluate_with_video.py \
  launch_multi_seed.py; do
  if [ -f "$REPO_ROOT/$f" ]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp "$REPO_ROOT/$f" "$BACKUP_DIR/$f"
  fi
  if [ -f "$SRC_ROOT/$f" ]; then
    mkdir -p "$REPO_ROOT/$(dirname "$f")"
    cp "$SRC_ROOT/$f" "$REPO_ROOT/$f"
  fi
done

chmod +x "$REPO_ROOT/launch_multi_seed.py" || true

echo "[INFO] Running syntax check..."
python -m py_compile \
  "$REPO_ROOT/train/common/config.py" \
  "$REPO_ROOT/train/common/callbacks.py" \
  "$REPO_ROOT/train/train_sac.py" \
  "$REPO_ROOT/utils/env_utils.py" \
  "$REPO_ROOT/utils/io_utils.py" \
  "$REPO_ROOT/evaluate/evaluate_with_video.py"

echo "[OK] Applied optimized training files."
echo "[OK] Backups saved to: $BACKUP_DIR"
