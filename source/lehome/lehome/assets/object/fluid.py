import os
import random
import torch
import numpy as np
import omni.kit.commands
import carb
from scipy.spatial import Delaunay
import open3d as o3d

from omni.physx.scripts import particleUtils, physicsUtils
try:
    from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
except ModuleNotFoundError:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.replicator.behavior")
    from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, delete_prim, get_prim_at_path
from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim, SingleXFormPrim

from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema
from omegaconf import DictConfig


class FluidObject:
    """
    FluidObject class for simulating fluid particles in Isaac Sim.
    Manages fluid particle systems, containers, materials, and physics properties.
    """

    def __init__(
        self,
        env_id: int,
        env_origin: torch.Tensor,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        use_container: bool = True,
    ):
        """
        Initialize the FluidObject with configuration, USD assets, and physics setup.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this fluid
            config: Configuration dictionary containing fluid properties
        """
        # Enable CPU fluid updates
        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", True)

        # --- 1. Configuration Parsing ---
        self.env_id = env_id
        self.env_origin = env_origin
        self.prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})

        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )
        self.global_common_config = self.global_config.objects.common

        self.physics_cfg = self.instance_config.get("physics", {})

        # --- 2. Prim and Path Initialization ---
        self.usd_prim_path = prim_path
        self.prim_name = self.instance_name
        self.env_name = prim_path_parts[-4]
        self.mesh_prim_path = self.usd_prim_path + "/water/water"
        self.stage = get_current_stage()

        # --- 3. Initial Pose and Asset Loading ---
        self.init_pos, self.init_ori, self.init_scale = self._get_initial_pose()
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # --- 4. Container Setup ---
        self.use_container = use_container
        self.container = None
        self.container_usd_path = None
        self.container_prim_path = None
        self.container_position = None
        self.container_orientation = None

        if self.use_container:
            self.container_usd_path = (
                os.getcwd() + "/Assets/objects/Fluids/Cup012/Cup012.usd"
            )
            self.container_prim_path = find_unique_string_name(
                initial_name=os.path.dirname(prim_path) + "/container",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            add_reference_to_stage(
                usd_path=self.container_usd_path, prim_path=self.container_prim_path
            )
            self.container_position = Gf.Vec3d(
                self.init_pos[0], self.init_pos[1], self.init_pos[2] - 0.08
            )
            self.container_orientation = self.init_ori
            self.container = SingleXFormPrim(
                prim_path=self.container_prim_path,
                name=f"fluid_container_{self.prim_name}",
                scale=self.instance_config.get("visual", {}).get(
                    "container_scale", [1.0, 1.0, 1.0]
                ),
            )

        # --- 5. Particle System Setup ---
        inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        interaction_flag = self.category_config.get("interaction_with_object", False)

        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/fluid_particle_system"
            )

        if not is_prim_path_valid(self.particle_system_path):
            self.particle_system = PhysxSchema.PhysxParticleSystem.Define(
                self.stage, self.particle_system_path
            )
        else:
            prim = self.stage.GetPrimAtPath(self.particle_system_path)
            self.particle_system = PhysxSchema.PhysxParticleSystem(prim)

        # Configure particle system properties
        self.particle_system.CreateParticleContactOffsetAttr().Set(
            inst_particle_system_cfg.get("particle_contact_offset", 0.025)
        )
        self.particle_system.CreateContactOffsetAttr().Set(
            inst_particle_system_cfg.get("contact_offset", 0.025)
        )
        self.particle_system.CreateRestOffsetAttr().Set(
            inst_particle_system_cfg.get("rest_offset", 0.0225)
        )
        self.particle_system.CreateFluidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        )
        self.particle_system.CreateSolidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("solid_rest_offset", 0.0225)
        )
        self.particle_system.CreateMaxVelocityAttr().Set(
            inst_particle_system_cfg.get("max_velocity", 2.5)
        )

        # Apply optional particle system APIs
        if inst_particle_system_cfg.get("smoothing", False):
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("anisotropy", False):
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("isosurface", True):
            PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.GetPrim())

        # --- 6. Particle Generation and Instancing ---
        fluid_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(self.mesh_prim_path))
        fluid_volumn_multiplier = self.physics_cfg.get("fluid_volumn", 1.0)
        cloud_points_base = np.array(fluid_mesh.GetPointsAttr().Get())
        cloud_points = cloud_points_base * fluid_volumn_multiplier
        fluid_rest_offset = inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        particleSpacing = 2.0 * fluid_rest_offset

        self.init_particle_positions, self.init_particle_velocities = (
            generate_particles_in_convex_mesh(
                vertices=cloud_points, sphere_diameter=particleSpacing, visualize=False
            )
        )
        self.stage.GetPrimAtPath(self.mesh_prim_path).SetActive(False)

        self.particle_point_instancer_path = Sdf.Path(self.usd_prim_path).AppendChild(
            "particles"
        )

        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage,
            path=self.particle_point_instancer_path,
            positions=Vt.Vec3fArray(self.init_particle_positions),
            velocities=Vt.Vec3fArray(self.init_particle_velocities),
            particle_system_path=self.particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.001,
            density=0.0,
        )

        self.point_instancer = UsdGeom.PointInstancer.Get(
            self.stage, self.particle_point_instancer_path
        )

        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f(
                float(self.init_pos[0]),
                float(self.init_pos[1]),
                float(self.init_pos[2]),
            ),
            orient=Gf.Quatf(
                float(self.init_ori[0]),
                Gf.Vec3f(
                    float(self.init_ori[1]),
                    float(self.init_ori[2]),
                    float(self.init_ori[3]),
                ),
            ),
            scale=Gf.Vec3f(
                float(self.init_scale[0]),
                float(self.init_scale[1]),
                float(self.init_scale[2]),
            ),
        )

        proto_path = self.particle_point_instancer_path.AppendChild(
            "particlePrototype0"
        )
        self.particle_prototype_path = proto_path  # Save path for later use

        particle_prototype_sphere = UsdGeom.Sphere.Get(self.stage, proto_path)
        particle_prototype_sphere.CreateRadiusAttr().Set(fluid_rest_offset)
        if inst_particle_system_cfg.get("isosurface", True):
            UsdGeom.Imageable(particle_prototype_sphere).MakeInvisible()

        # --- 7. Initial Material and Physics Properties Setup ---
        self._apply_random_material()

        # self._handle_semantic_labels()

    def _apply_random_material(self):
        """
        Selects a random material from configuration, creates it, and binds it
        to the particle system and particle prototypes. Also sets physics properties.
        """
        # Select a random material URL from config
        visual_cfg = self.instance_config.get("visual", {})
        material_cfg = visual_cfg.get("visual_material", {})
        material_list = material_cfg.get("material_usd_folder", [])

        if material_list:
            material_url = random.choice(material_list)
            project_root = os.getcwd()
            material_url = os.path.abspath(os.path.join(project_root, material_url))
        else:
            material_url = "./Assets/Material/Base/Textiles/Linen_Blue.mdl"

        # Create the material prim with unique name
        material_name = os.path.splitext(os.path.basename(material_url))[0]
        if is_prim_path_valid(f"{os.path.dirname(self.prim_path)}/Looks/material"):
            delete_prim(f"{os.path.dirname(self.prim_path)}/Looks/material")

        unique_material_name = find_unique_string_name(
            initial_name=f"{os.path.dirname(self.prim_path)}/Looks/material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        color_material_path = unique_material_name
        create_mdl_material(material_url, material_name, color_material_path)

        # Set physics material properties
        inst_particle_material_cfg = self.physics_cfg.get("particle_material", {})
        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=color_material_path,
            adhesion=inst_particle_material_cfg.get("adhesion"),
            adhesion_offset_scale=inst_particle_material_cfg.get(
                "adhesion_offset_scale"
            ),
            cohesion=inst_particle_material_cfg.get("cohesion"),
            particle_adhesion_scale=inst_particle_material_cfg.get(
                "particle_adhesion_scale"
            ),
            particle_friction_scale=inst_particle_material_cfg.get(
                "particle_friction_scale"
            ),
            drag=inst_particle_material_cfg.get("drag"),
            lift=inst_particle_material_cfg.get("lift"),
            friction=inst_particle_material_cfg.get("friction"),
            damping=inst_particle_material_cfg.get("damping"),
            gravity_scale=inst_particle_material_cfg.get("gravity_scale", 1.0),
            viscosity=inst_particle_material_cfg.get("viscosity"),
            vorticity_confinement=inst_particle_material_cfg.get(
                "vorticity_confinement"
            ),
            surface_tension=inst_particle_material_cfg.get("surface_tension"),
            density=inst_particle_material_cfg.get("density"),
            cfl_coefficient=inst_particle_material_cfg.get("cfl_coefficient"),
        )

        # Bind material to particle system and prototype
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.particle_system_path,
            material_path=color_material_path,
        )

        if hasattr(self, "particle_prototype_path") and is_prim_path_valid(
            self.particle_prototype_path
        ):
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.particle_prototype_path,
                material_path=color_material_path,
            )

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def initialize(self):
        """Initialize the fluid container position."""
        if self.container is not None:
            self.container.set_local_pose(
                translation=self.container_position,
                orientation=self.container_orientation,
            )

    def reset(self, soft=False):
        """
        Reset the fluid system to initial state with new position and orientation.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """
        # Re-apply random material
        self._apply_random_material()

        instance_config = self.instance_config
        common_config = self.global_config.objects.common

        # Determine new position of water
        if "pos" in instance_config:
            water_pos = instance_config.pos
        else:
            pos_range = (
                common_config.soft_reset_pos_range
                if soft
                else common_config.initial_pos_range
            )
            water_pos = [
                random.uniform(pos_range[0], pos_range[3]),
                random.uniform(pos_range[1], pos_range[4]),
                random.uniform(pos_range[2], pos_range[5]),
            ]

        if "ori" in instance_config:
            ori_euler = instance_config.ori
        else:
            ori_range = (
                common_config.soft_reset_ori_range
                if soft
                else common_config.initial_ori_range
            )
            ori_euler = [
                random.uniform(ori_range[0], ori_range[3]),
                random.uniform(ori_range[1], ori_range[4]),
                random.uniform(ori_range[2], ori_range[5]),
            ]
        ori_quat = euler_angles_to_quat(ori_euler, degrees=True)

        scale = self.instance_config.get("visual", {}).get(
            "container_scale", [1.0, 1.0, 1.0]
        )
        if self.use_container:
            cup_pos = [water_pos[0], water_pos[1], water_pos[2] - 0.08]
            delete_prim(self.container_prim_path)
            add_reference_to_stage(
                usd_path=self.container_usd_path, prim_path=self.container_prim_path
            )
            self.container = SingleXFormPrim(
                prim_path=self.container_prim_path,
                name=f"fluid_container_{self.prim_name}",
                scale=scale,
            )
            self.container.set_local_pose(translation=cup_pos, orientation=ori_quat)

        # save reset pose of water
        self.reset_pose = np.concatenate(
            [
                np.array(water_pos, dtype=np.float32),
                np.array(ori_quat, dtype=np.float32),
            ]
        )

        # 5) reset water particle
        self.set_particle_positions(self.init_particle_positions)
        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f(
                float(water_pos[0]),
                float(water_pos[1]),
                float(water_pos[2]),
            ),
            orient=Gf.Quatf(
                float(ori_quat[0]),
                Gf.Vec3f(
                    float(ori_quat[1]),
                    float(ori_quat[2]),
                    float(ori_quat[3]),
                ),
            ),
            scale=Gf.Vec3f(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
            ),
        )

    def get_particle_positions(
        self, visualize: bool = True, global_coord: bool = False
    ):
        """
        Get current positions of all fluid particles.

        Args:
            visualize: Whether to visualize particles using Open3D

        Returns:
            positions: Array of particle positions
        """
        positions = np.array(
            self.point_instancer.GetPositionsAttr().Get(), dtype=np.float32
        )

        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            o3d.visualization.draw_geometries([pcd])

        if global_coord:
            positions = positions + self.env_origin.detach().numpy()
        return positions, None, None

    def set_particle_positions(self, positions: np.ndarray, global_coord: bool = False):
        """
        Set positions of all fluid particles.

        Args:
            positions: Array of new particle positions
        """
        positions_vt = Vt.Vec3fArray(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        )

        if global_coord:
            positions = positions - self.env_origin.detach().numpy()
            positions_vt = Vt.Vec3fArray(
                [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
            )

        self.point_instancer.GetPositionsAttr().Set(positions_vt)

    def _get_initial_pose(self):
        """
        Determine initial position, orientation, and scale from configuration.

        Returns:
            Initial position, orientation (quaternion), and scale
        """
        # Determine initial position
        if "pos" in self.instance_config and self.instance_config["pos"] is not None:
            pos = self.instance_config["pos"]
        else:
            pos_range = self.category_common_config.get(
                "initial_pos_range", self.global_common_config.initial_pos_range
            )
            pos = [
                random.uniform(pos_range[0], pos_range[3]),
                random.uniform(pos_range[1], pos_range[4]),
                random.uniform(pos_range[2], pos_range[5]),
            ]

        # Determine initial orientation
        if "ori" in self.instance_config and self.instance_config["ori"] is not None:
            ori_euler = self.instance_config["ori"]
        else:
            ori_range = self.category_common_config.get(
                "initial_ori_range", self.global_common_config.initial_ori_range
            )
            ori_euler = [
                random.uniform(ori_range[0], ori_range[3]),
                random.uniform(ori_range[1], ori_range[4]),
                random.uniform(ori_range[2], ori_range[5]),
            ]

        # Determine scale
        instance_common_config = self.instance_config.get("common", {})
        instance_scale = instance_common_config.get("scale")
        if instance_scale is not None:
            scale = instance_scale
        else:
            scale = self.global_common_config.scale
        quat = euler_angles_to_quat(ori_euler, degrees=True)
        self.reset_pose = np.concatenate(
            [np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32)]
        )
        return (
            np.array(pos),
            quat,
            np.array(scale),
        )

    def get_pose_data(self) -> dict:
        """Return pose as unified nested dict (no key) with quaternion rotation.

        Returns:
            dict with keys 'trans' [x,y,z], 'rot' [w,x,y,z], 'scale' [sx,sy,sz].
        """
        pos = self.reset_pose[:3].tolist()
        quat = self.reset_pose[3:].tolist()
        return {"trans": pos, "rot": quat, "scale": list(self.init_scale)}

    def get_all_pose(self):
        """Return pose dict keyed by 'Cup'. Delegates to get_pose_data()."""
        return {"Cup": self.get_pose_data()}

    def set_pose_from_data(self, data: dict):
        """Restore pose from unified data dict (no key).

        Args:
            data: dict with keys 'trans' [x,y,z] and 'rot' [w,x,y,z] (quaternion).
        """
        pos = np.array(data["trans"], dtype=np.float32)
        quat = np.array(data["rot"], dtype=np.float32)  # already quaternion
        
        if self.container is not None:
            cup_pos = [pos[0], pos[1], pos[2] - 0.08]
            self.container.set_local_pose(translation=cup_pos, orientation=quat)
        
        from omni.physx.scripts import physicsUtils
        from pxr import Gf
        
        scale = getattr(self, "init_scale", [1.0, 1.0, 1.0])
        self.set_particle_positions(self.init_particle_positions)
        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
            orient=Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))),
            scale=Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])),
        )
        
        # Update reset_pose to store 7 elements
        self.reset_pose = np.concatenate([pos, quat])

    def set_all_pose(self, pose_dict: dict):
        """Restore pose from keyed dict. Expects key 'Cup'."""
        if "Cup" in pose_dict:
            self.set_pose_from_data(pose_dict["Cup"])



def generate_particles_in_convex_mesh(
    vertices: np.ndarray, sphere_diameter: float, visualize: bool = False
):
    """
    Generate particles within a convex mesh using Delaunay triangulation.

    Args:
        vertices: Vertices of the convex mesh
        sphere_diameter: Diameter of particles to generate
        visualize: Whether to visualize the particles and mesh vertices

    Returns:
        List of particle positions and velocities (zero-initialized)
    """
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    hull = Delaunay(vertices)

    # Create grid of sample points
    x_vals = np.arange(min_bound[0], max_bound[0], sphere_diameter)
    y_vals = np.arange(min_bound[1], max_bound[1], sphere_diameter)
    z_vals = np.arange(min_bound[2], max_bound[2], sphere_diameter)

    samples = np.stack(
        np.meshgrid(x_vals, y_vals, z_vals, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    # Find points inside the convex hull
    inside_mask = hull.find_simplex(samples) >= 0
    inside_points = samples[inside_mask]

    # Initialize velocities to zero
    velocity = np.zeros_like(inside_points)

    # Visualization
    if visualize:
        particle_pcd = o3d.geometry.PointCloud()
        particle_pcd.points = o3d.utility.Vector3dVector(inside_points)
        particle_pcd.paint_uniform_color([0.2, 0.4, 1.0])

        vertex_pcd = o3d.geometry.PointCloud()
        vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
        vertex_pcd.paint_uniform_color([1.0, 0.1, 0.1])

        o3d.visualization.draw_geometries(
            [particle_pcd, vertex_pcd], window_name="Convex Mesh Particle Filling"
        )

    return [Gf.Vec3f(*pt) for pt in inside_points], [Gf.Vec3f(*vel) for vel in velocity]
