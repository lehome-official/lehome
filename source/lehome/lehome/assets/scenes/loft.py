from pathlib import Path

from lehome.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Loft Scene"""
ASSETS_ROOT_PATH = Path(ASSETS_ROOT)

LW_LOFT_USD_PATH = str(ASSETS_ROOT_PATH / "LW_Loft" / "LW_Loft_bedroom.usd")
LEKIWI_LOFT_USD_PATH = str(
    ASSETS_ROOT_PATH / "scenes" / "kitchen_with_orange" / "scene.usd"
)
# Dedicated scene path for xlerobot teleoperation.
XLEROBOT_LOFT_USD_PATH = LEKIWI_LOFT_USD_PATH

LW_LOFT_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LW_LOFT_USD_PATH,
    )
)
