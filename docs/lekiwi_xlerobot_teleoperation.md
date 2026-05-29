# Lekiwi and Xlerobot Teleoperation

This guide covers the Lekiwi and Xlerobot teleoperation examples.

## Assets

Download the release assets into the repository root:

```bash
hf download lehome/lehome_release --repo-type dataset --local-dir Assets
```

The teleoperation environments use these asset paths:

| Robot | Asset |
|---|---|
| Lekiwi | `Assets/lekiwi/lekiwi_final_best.usd` |
| Xlerobot | `Assets/robots/xlerobot/xlerobot/xlerobot_final_best.usd` |
| Scene | `Assets/scenes/kitchen_with_orange/scene.usd` |

## Keyboard Control

Run from the repository root after activating the LeHome Python environment.

```bash
python scripts/teleoperation/teleop_lekiwi.py --device=cuda --enable_cameras
python scripts/teleoperation/teleop_xlerobot.py --device=cuda --enable_cameras
```

Press `B` in the simulator window to start control. Use `W/A/S/D` and `Q/E` for mobile-base movement. Use `F5` to reset the environment.

## Hybrid Control

Hybrid control uses the keyboard for the mobile base and SO101 leader arms for arm control. Grant serial access before starting:

```bash
sudo chmod 666 /dev/ttyACM*
```

Lekiwi hybrid:

```bash
python scripts/teleoperation/teleop_lekiwi_hybrid.py --device=cuda --enable_cameras --arm_port=/dev/ttyACM0 --control_mode=hybrid
```

Xlerobot hybrid:

```bash
python scripts/teleoperation/teleop_xlerobot_hybrid.py --device=cuda --enable_cameras --left_arm_port=/dev/ttyACM0 --right_arm_port=/dev/ttyACM0 --control_mode=hybrid
```

Use `F6` to switch hybrid control modes. Add `--quiet_kit_logs` to reduce Omniverse and PhysX startup log noise.
