#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_FILES = [
    "Assets/lekiwi/lekiwi_final_best.usd",
    "Assets/lekiwi/test_lekiwi.usd",
    "Assets/scenes/1BRAPT_LeHome/1BRAPT_LeHome.usd",
    "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_002/TCLC_002_obj.usd",
    "Assets/Material/Garment/linen_Blue.usd",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check required local assets for lekiwi teleoperation")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: auto-detected)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    missing: list[str] = []

    print(f"[check] repo root: {root}")
    for rel in REQUIRED_FILES:
        abs_path = root / rel
        if abs_path.exists():
            size_kib = abs_path.stat().st_size / 1024.0
            print(f"[ok]   {rel} ({size_kib:.1f} KiB)")
        else:
            print(f"[miss] {rel}")
            missing.append(rel)

    if missing:
        print("[check] result: FAILED")
        print("[check] missing files:")
        for rel in missing:
            print(f"  - {rel}")
        return 1

    print("[check] result: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
