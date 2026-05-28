#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[xlerobot-check] repo: $REPO_ROOT"

echo "[xlerobot-check] compileall..."
python -m compileall \
  source/lehome/lehome/tasks/xlerobot_task \
  source/lehome/lehome/assets/robots/xlerobot.py \
  source/lehome/lehome/devices/keyboard/xlerobot_keyboard.py \
  source/lehome/lehome/devices/hybrid/xlerobot_hybrid_controller.py \
  source/lehome/lehome/devices/lerobot/xlerobot_leader.py \
  source/lehome/lehome/devices/lerobot/bi_xlerobot_leader.py \
  source/lehome/lehome/devices/xlerobot_action_process.py \
  scripts/teleoperation/teleop_xlerobot.py \
  scripts/teleoperation/teleop_xlerobot_hybrid.py \
  scripts/tools/inspect_xlerobot_usd.py >/dev/null

echo "[xlerobot-check] teleop num_envs wiring..."
for f in scripts/teleoperation/teleop_xlerobot.py scripts/teleoperation/teleop_xlerobot_hybrid.py; do
  if ! rg -n 'num_envs=args_cli\.num_envs' "$f" >/dev/null; then
    echo "ERROR: $f missing num_envs pass-through."
    exit 1
  fi
done

echo "[xlerobot-check] garment cfg uniqueness..."
cfg_count="$(find source/lehome/lehome/tasks/xlerobot_task -name 'particle_garment_cfg.yaml' | wc -l | tr -d ' ')"
if [[ "$cfg_count" != "1" ]]; then
  echo "ERROR: expected exactly 1 xlerobot particle_garment_cfg.yaml, found $cfg_count."
  find source/lehome/lehome/tasks/xlerobot_task -name 'particle_garment_cfg.yaml' -print
  exit 1
fi

echo "[xlerobot-check] OK"
