#!/usr/bin/env python3
"""Inspect xlerobot USD mobile-base joints for migration/debug validation.

Usage:
  python scripts/tools/inspect_xlerobot_usd.py
  python scripts/tools/inspect_xlerobot_usd.py --usd Assets/robots/xlerobot/xlerobot/xlerobot_final_best.usd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from builtins import print as _builtin_print


def print(*args, **kwargs):
    """Flush immediately to survive SimulationApp fast shutdown."""
    kwargs.setdefault("flush", True)
    return _builtin_print(*args, **kwargs)


TARGET_JOINTS = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect xlerobot mobile-base joints in USD")
    parser.add_argument(
        "--usd",
        type=str,
        default="Assets/robots/xlerobot/xlerobot/xlerobot_final_best.usd",
        help="Path to xlerobot USD file",
    )
    return parser


def _print_joint_info(stage, prim, UsdPhysics):
    name = prim.GetName()
    print(f"\n=== {name} ===")
    print(f"path      : {prim.GetPath()}")
    print(f"type      : {prim.GetTypeName()}")

    joint = UsdPhysics.Joint(prim)
    body0 = [str(x) for x in joint.GetBody0Rel().GetTargets()]
    body1 = [str(x) for x in joint.GetBody1Rel().GetTargets()]
    print(f"body0     : {body0}")
    print(f"body1     : {body1}")
    print(f"localPos0 : {joint.GetLocalPos0Attr().Get()}")
    print(f"localPos1 : {joint.GetLocalPos1Attr().Get()}")
    print(f"localRot0 : {joint.GetLocalRot0Attr().Get()}")
    print(f"localRot1 : {joint.GetLocalRot1Attr().Get()}")

    if prim.IsA(UsdPhysics.PrismaticJoint):
        pj = UsdPhysics.PrismaticJoint(prim)
        print(f"axis      : {pj.GetAxisAttr().Get()}")
        print(f"limits    : [{pj.GetLowerLimitAttr().Get()}, {pj.GetUpperLimitAttr().Get()}]")
    elif prim.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(prim)
        print(f"axis      : {rj.GetAxisAttr().Get()}")
        print(f"limits    : [{rj.GetLowerLimitAttr().Get()}, {rj.GetUpperLimitAttr().Get()}]")

    print("drives     :")
    drive_found = False
    for dof_name in ("linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Get(prim, dof_name)
        if drive and drive.GetPrim().IsValid() and drive.GetTypeAttr().HasAuthoredValueOpinion():
            drive_found = True
            print(
                f"  - {dof_name}: "
                f"type={drive.GetTypeAttr().Get()} "
                f"stiff={drive.GetStiffnessAttr().Get()} "
                f"damp={drive.GetDampingAttr().Get()} "
                f"maxF={drive.GetMaxForceAttr().Get()} "
                f"targetPos={drive.GetTargetPositionAttr().Get()} "
                f"targetVel={drive.GetTargetVelocityAttr().Get()}"
            )
    if not drive_found:
        print("  - <none>")


def main() -> int:
    args = _create_parser().parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    if not usd_path.exists():
        print(f"ERROR: USD file not found: {usd_path}")
        return 1

    try:
        from isaacsim import SimulationApp
    except Exception as exc:
        print("ERROR: cannot import isaacsim.SimulationApp. Please run in Isaac Sim Python env.")
        print(f"detail: {exc}")
        return 1

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            print(f"ERROR: failed to open USD stage: {usd_path}")
            return 1

        print(f"USD: {usd_path}")
        default_prim = stage.GetDefaultPrim()
        print(f"default_prim: {default_prim.GetPath() if default_prim else '<none>'}")

        articulation_roots = [
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        print(f"articulation_roots: {articulation_roots}")

        found = []
        for joint_name in TARGET_JOINTS:
            joint_prim = None
            for prim in stage.Traverse():
                if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint):
                    joint_prim = prim
                    break
            if joint_prim is None:
                print(f"\n=== {joint_name} ===")
                print("MISSING")
                continue

            found.append(joint_name)
            _print_joint_info(stage, joint_prim, UsdPhysics)

        print("\nSummary:")
        print(f"  expected: {list(TARGET_JOINTS)}")
        print(f"  found   : {found}")
        missing = [name for name in TARGET_JOINTS if name not in found]
        print(f"  missing : {missing}")

        return 0 if not missing else 2
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
