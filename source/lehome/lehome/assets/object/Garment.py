import torch
import numpy as np
import random
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem, SingleXFormPrim
from isaacsim.core.api.materials.particle_material import ParticleMaterial as _ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Vt
import os

# from ..infinigen_sdg_utils import get_usd_paths_from_folder
from omegaconf import DictConfig
from termcolor import cprint


class ParticleMaterial(_ParticleMaterial):
    """ParticleMaterial fallback for IsaacLab's patched SimulationManager."""

    def __init__(self, *args, **kwargs):
        self._backend = SimulationManager.get_backend()
        self._device = SimulationManager.get_physics_sim_device()
        self._backend_utils = SimulationManager._get_backend_utils()
        super().__init__(*args, **kwargs)


class GarmentObject(SingleClothPrim):
    """
    GarmentObject class that wraps the Isaac Sim SingleCloth prim functionality.
    This class inherits from the Isaac Sim SingleClothPrim class and can be extended
    """

    @staticmethod
    def _core_pose(value):
        """Convert pose values for Isaac Sim core prim wrappers."""
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def set_world_pose(self, position=None, orientation=None):
        """Set world pose using the wrapped prim view's backend."""
        if position is not None:
            position = self._prim_view._backend_utils.convert(
                self._core_pose(position), device=self._prim_view._device
            )
            position = self._prim_view._backend_utils.expand_dims(position, 0)
        if orientation is not None:
            orientation = self._prim_view._backend_utils.convert(
                self._core_pose(orientation), device=self._prim_view._device
            )
            orientation = self._prim_view._backend_utils.expand_dims(orientation, 0)
        self._prim_view.set_world_poses(positions=position, orientations=orientation)

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        visual_usd_path: str,
        config: DictConfig,
    ):
        """
        Initialize the GarmentObject with position, orientation, and configuration.

        Args:
            prim_path: Path to the prim in the stage.
            usd_path: Path to the USD asset file for this object.
            visual_material_usd_path: Path to the USD asset file for the visual material.
            config: Configuration dictionary containing object properties.

            1. set pos and ori for garment object
            2. create physics material and visual material for garment object
            3. randomize physics material and visual material according to the config
            4. record flat state for the garment object
        """
        # -------- Parameters Configuration ---------#
        # usd prim path
        self.usd_prim_path = prim_path
        cprint(f"usd prim path: {self.usd_prim_path}", "green")
        # usd name
        self.prim_name = prim_path.split("/")[-1]
        # mesh prim path which is contained in the usd asset
        self.mesh_prim_path = find_unique_string_name(
            self.usd_prim_path + "/mesh",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        # particle system path
        self.particle_system_path = find_unique_string_name(
            self.usd_prim_path + "/particle_system",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        # particle material path
        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.visual_usd_path = visual_usd_path
        # garment configuration
        self.config = config
        self.objects_config = config.get("objects")
        # world prim encapsulation
        self.world_prim = SingleXFormPrim(self.usd_prim_path)

        # -------- Loading Procedure ---------#
        # get initial state
        self.init_pos, self.init_ori, self.init_scale = self._get_initial_pose()
        # Load USD asset as a reference
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        # define particle system for garment
        self.particle_system = SingleParticleSystem(
            prim_path=self.particle_system_path,
            particle_system_enabled=self.objects_config.particle_system.get(
                "particle_system_enabled", None
            ),
            enable_ccd=self.objects_config.particle_system.get("enable_ccd", None),
            solver_position_iteration_count=self.objects_config.particle_system.get(
                "solver_position_iteration_count", None
            ),
            max_depenetration_velocity=self.objects_config.particle_system.get(
                "max_depenetration_velocity", None
            ),
            global_self_collision_enabled=self.objects_config.particle_system.get(
                "global_self_collision_enabled", None
            ),
            non_particle_collision_enabled=self.objects_config.particle_system.get(
                "non_particle_collision_enabled", None
            ),
            contact_offset=self.objects_config.particle_system.get(
                "contact_offset", None
            ),
            rest_offset=self.objects_config.particle_system.get("rest_offset", None),
            particle_contact_offset=self.objects_config.particle_system.get(
                "particle_contact_offset", None
            ),
            fluid_rest_offset=self.objects_config.particle_system.get(
                "fluid_rest_offset", None
            ),
            solid_rest_offset=self.objects_config.particle_system.get(
                "solid_rest_offset", None
            ),
            wind=self.objects_config.particle_system.get("wind", None),
            max_neighborhood=self.objects_config.particle_system.get(
                "max_neighborhood", None
            ),
            max_velocity=self.objects_config.particle_system.get("max_velocity", None),
        )
        # define particle material for garment
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            adhesion=self.objects_config.particle_material.get("adhesion", None),
            adhesion_offset_scale=self.objects_config.particle_material.get(
                "adhesion_offset_scale", None
            ),
            cohesion=self.objects_config.particle_material.get("cohesion", None),
            particle_adhesion_scale=self.objects_config.particle_material.get(
                "particle_adhesion_scale", None
            ),
            particle_friction_scale=self.objects_config.particle_material.get(
                "particle_friction_scale", None
            ),
            drag=self.objects_config.particle_material.get("drag", None),
            lift=self.objects_config.particle_material.get("lift", None),
            friction=self.objects_config.particle_material.get("friction", None),
            damping=self.objects_config.particle_material.get("damping", None),
            gravity_scale=self.objects_config.particle_material.get(
                "gravity_scale", None
            ),
            viscosity=self.objects_config.particle_material.get("viscosity", None),
            vorticity_confinement=self.objects_config.particle_material.get(
                "vorticity_confinement", None
            ),
            surface_tension=self.objects_config.particle_material.get(
                "surface_tension", None
            ),
        )
        self.num_count = 0
        # add particle cloth attribute to garment
        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=self.objects_config.garment_config.get("particle_mass", None),
            self_collision=self.objects_config.garment_config.get(
                "self_collision", None
            ),
            self_collision_filter=self.objects_config.garment_config.get(
                "self_collision_filter", None
            ),
            stretch_stiffness=self.objects_config.garment_config.get(
                "stretch_stiffness", None
            ),
            bend_stiffness=self.objects_config.garment_config.get(
                "bend_stiffness", None
            ),
            shear_stiffness=self.objects_config.garment_config.get(
                "shear_stiffness", None
            ),
            spring_damping=self.objects_config.garment_config.get(
                "spring_damping", None
            ),
        )

        # set visual material
        if self.visual_usd_path is not None:
            self._apply_visual_material(self.visual_usd_path)
        self.set_world_pose(
            position=self._core_pose(self.init_pos),
            orientation=self._core_pose(self.init_ori),
        )

    def initialize(self):
        """
        Initialize the object by setting its initial position and orientation,
        while also get initial info of particles that make up the object.
        """
        # set local pose for initialization (wait for the update of scene manager)
        self.set_world_pose(
            position=self._core_pose(self.init_pos),
            orientation=self._core_pose(self.init_ori),
        )
        if "cuda" in self._device:
            self.physics_sim_view = SimulationManager.get_physics_sim_view()
            self._cloth_prim_view.initialize(self.physics_sim_view)

        # get initial info of particles that make up the object
        self._get_initial_info()

        self._prim.GetAttribute("points").Set(
            Vt.Vec3fArray.FromNumpy(self._get_points_pose().detach().cpu().numpy())
        )

    def _restore_initial_particles(self):
        """Restore the cached cloth geometry before applying a new world pose.

        We first move the parent xform back to the identity pose so the cached
        particle positions are not re-applied under a stale transform.
        """
        identity_pos = [0.0, 0.0, 0.0]
        identity_quat = euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True)
        self.set_world_pose(
            position=self._core_pose(identity_pos),
            orientation=self._core_pose(identity_quat),
        )

        self._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(self.initial_points_positions))

    def reset(self):
        """
        Perform soft reset by randomly modifying the object's position and orientation.
        Meanwhile,return back to the initial positions of all particles that make up the object.
        """
        self._restore_initial_particles()
        # Get position range from configuration
        pos_reset_range = self.objects_config.common.soft_reset_pos_range
        rot_reset_range = self.objects_config.common.soft_reset_rot_range
        # Generate random initial position/rotation within range
        pos = [
            random.uniform(pos_reset_range[0], pos_reset_range[3]),
            random.uniform(pos_reset_range[1], pos_reset_range[4]),
            random.uniform(pos_reset_range[2], pos_reset_range[5]),
        ]
        ori = [
            random.uniform(rot_reset_range[0], rot_reset_range[3]),
            random.uniform(rot_reset_range[1], rot_reset_range[4]),
            random.uniform(rot_reset_range[2], rot_reset_range[5]),
        ]
        quat = euler_angles_to_quat(ori, degrees=True)
        self.set_world_pose(position=self._core_pose(pos), orientation=self._core_pose(quat))
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32)]
        )

    def get_current_mesh_points(
        self, visualize=False, save=False, save_path="./pointcloud.ply"
    ):
        """
        Get the current mesh points of the garment.
        Input:
            visualize: whether to visualize the mesh points
            save: whether to save the mesh points
            save_path: the path to save the mesh points
        Output:
            transformed_points: the current transformed mesh points of the garment, which is used for actual visualization
            mesh_points: the current original mesh points of the garment
            pos_world: the current world position of the garment (This parameter is suitable for cpu version, which will be set to None in gpu version)
            ori_world: the current world orientation of the garment (This parameter is suitable for cpu version, which will be set to None in gpu version)
        """
        if self._device == "cpu":
            pos_world, ori_world = self.get_world_pose()
            scale_world = self.get_world_scale()
            mesh_points = self._get_points_pose().detach().cpu().numpy()
            transformed_mesh_points = self.transform_points(
                mesh_points,
                pos_world.detach().cpu().numpy(),
                ori_world.detach().cpu().numpy(),
                scale_world.detach().cpu().numpy(),
            )
        else:
            mesh_points = (
                self._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            transformed_mesh_points = mesh_points
            pos_world = None
            ori_world = None
        # visualize the initial points
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)
        if visualize:
            o3d.visualization.draw_geometries([pcd])
        if save:
            o3d.io.write_point_cloud(save_path, pcd)
            cprint(f"points saved to {save_path}", "green")
        return transformed_mesh_points, mesh_points, pos_world, ori_world

    def set_current_mesh_points(self, mesh_points, pos_world, ori_world):
        """
        Set the current mesh points of the deformable object.
        Input:
            mesh_points (ndarray): original mesh points, which is provided in 'get_current_mesh_points' function
            pos_world (ndarray): world position of the mesh points, which is provided in 'get_current_mesh_points' function, only need for cpu version
            ori_world (ndarray): world orientation of the mesh points, which is provided in 'get_current_mesh_points' function, only need for cpu version
        """
        if self._device == "cpu":
            if pos_world is None or ori_world is None:
                raise ValueError(
                    "pos_world and ori_world must be provided if device is cpu"
            )
            self._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(mesh_points))
            # self.set_world_pose(pos_world, ori_world)
            self.world_prim.set_world_pose(self._core_pose(pos_world), self._core_pose(ori_world))
        else:
            current_mesh_points = (
                torch.from_numpy(mesh_points).to(self._device).unsqueeze(0)
            )
            self._cloth_prim_view.set_world_positions(current_mesh_points)
        return

    def _apply_visual_material(self, material_path: str):
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        add_reference_to_stage(
            usd_path=material_path, prim_path=self.visual_material_path
        )
        self.visual_material_prim = prims_utils.get_prim_at_path(
            self.visual_material_path
        )
        self.material_prim = prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        self.mesh_prim = prims_utils.get_prim_at_path(self.mesh_prim_path)
        self.garment_submesh = prims_utils.get_prim_children(self.mesh_prim)
        if len(self.garment_submesh) == 0:
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,
                material_path=self.material_prim_path,
            )
        else:
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.mesh_prim_path,
                material_path=self.material_prim_path,
            )
            for prim in self.garment_submesh:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=self.material_prim_path,
                )

    def _get_initial_pose(self):
        """
        Get the initial pose (/ori) of the garment object.
        """
        # Get position range from configuration
        pos_init_range = self.objects_config.common.initial_pos_range
        rot_init_range = self.objects_config.common.initial_rot_range
        # Generate random initial position/rotation within range
        pos = [
            random.uniform(pos_init_range[0], pos_init_range[3]),
            random.uniform(pos_init_range[1], pos_init_range[4]),
            random.uniform(pos_init_range[2], pos_init_range[5]),
        ]
        ori = [
            random.uniform(rot_init_range[0], rot_init_range[3]),
            random.uniform(rot_init_range[1], rot_init_range[4]),
            random.uniform(rot_init_range[2], rot_init_range[5]),
        ]
        scale = self.objects_config.common.scale
        quat = euler_angles_to_quat(ori, degrees=True)
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32)]
        )
        # Set initial pose
        return pos, quat, scale

    def _get_initial_info(self):
        """
        Return the initial positions of all particles that make up the object.
        """
        self.initial_points_positions = self._get_points_pose().detach().cpu().numpy()

    def transform_points(self, points, pos, ori, scale):
        """
        Transform points by pos, ori and scale

        Args:
            points (numpy.ndarray): (N, 3) points to be transformed
            pos (numpy.ndarray): (3,) position transformation of the object
            ori (numpy.ndarray): (4,) orientation transformation of the object (quaternion)
            scale (int): scale transformation of the object
        """
        ori_matrix = quat_to_rot_matrix(ori)
        scaled_points = points * scale
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        """
        Inverse transform: Recover original points from transformed ones using pos, ori, and scale.

        Args:
            transformed_points (numpy.ndarray): (N, 3) transformed points in world space
            pos (numpy.ndarray): (3,) position transformation of the object
            ori (numpy.ndarray): (4,) orientation transformation of the object (quaternion, xyzw)
            scale (float): scale transformation of the object

        Returns:
            numpy.ndarray: (N, 3) original local-space points
        """
        ori_matrix = quat_to_rot_matrix(ori)
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        original_points = rotated_points / scale
        return original_points


    def get_pose_data(self) -> dict:
        """Return pose as unified nested dict (no key) with quaternion rotation.

        Returns:
            dict with keys 'trans' [x,y,z], 'rot' [w,x,y,z], 'scale' [sx,sy,sz].
        """
        pos = self.reset_pose[:3].tolist()
        quat = self.reset_pose[3:].tolist()
        return {"trans": pos, "rot": quat, "scale": list(self.init_scale)}

    def get_all_pose(self):
        """Return pose dict keyed by object name. Delegates to get_pose_data()."""
        return {"Garment": self.get_pose_data()}

    def set_pose_from_data(self, data: dict):
        """Restore pose from unified data dict (no key).

        Args:
            data: dict with keys 'trans' [x,y,z] and 'rot' [w,x,y,z] (quaternion).
        """
        pos = np.array(data["trans"], dtype=np.float32)
        quat = np.array(data["rot"], dtype=np.float32)  # already quaternion
        self._restore_initial_particles()
        self.set_world_pose(position=self._core_pose(pos), orientation=self._core_pose(quat))
        # Update reset_pose to store 7 elements
        self.reset_pose = np.concatenate([pos, quat])

    def set_all_pose(self, pose_dict: dict):
        """Restore pose from keyed dict. Expects key 'Garment'."""
        if "Garment" in pose_dict:
            self.set_pose_from_data(pose_dict["Garment"])
