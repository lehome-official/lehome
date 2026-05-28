# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="LeIsaac-Xlerobot-Direct-Task-v0",
    entry_point=f"{__name__}.xlerobot_env:XlerobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xlerobot_cfg:XlerobotEnvCfg",
    },
)

gym.register(
    id="LeHome-Xlerobot-Direct-Task-v0",
    entry_point=f"{__name__}.xlerobot_env:XlerobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xlerobot_cfg:XlerobotEnvCfg",
    },
)
