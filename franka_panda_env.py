# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
from typing import List, Literal, Optional

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *


def get_center_env_index(num_envs: int, num_per_row: int) -> int:
    """Get the index of the center environment.

    Args:
        num_envs (int): Number of environments.
        num_per_row (int): Number of environments per row.

    Returns:
        int: Index of the center environment.
    """
    num_rows = math.ceil(num_envs / num_per_row)

    center_row_index = num_rows // 2
    center_column_index = num_per_row // 2

    center_env_index = center_row_index * num_per_row + center_column_index

    return center_env_index


class FrankaPandaEnv:
    """Franka Panda environment."""

    def __init__(
        self,
        num_envs: int = 36,
        use_gpu: bool = True,
        use_gpu_pipeline: bool = False,
        num_threads: int = 1,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        headless: bool = False,
        cube_size: float = 0.05,
    ):
        """Initialize the Franka Panda environment.

        Args:
            num_envs (int, optional): Number of environments to create. Defaults to 36.
            use_gpu (bool, optional): Whether to use GPU. Defaults to True.
            use_gpu_pipeline (bool, optional): Whether to use GPU pipeline. Defaults to False.
            num_threads (int, optional): Number of threads. Defaults to 1.
            compute_device_id (int, optional): Compute device ID. Defaults to 0.
            graphics_device_id (int, optional): Graphics device ID. Defaults to 0.
            headless (bool, optional): Whether to run in headless mode. Defaults to False.
            cube_size (float, optional): Size of the cube. Defaults to 0.05.
        """
        self.gym = gymapi.acquire_gym()
        self.num_envs = num_envs
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.compute_device_id = compute_device_id
        self.graphics_device_id = graphics_device_id
        self.use_gpu_pipeline = use_gpu_pipeline
        self.device = None
        self.viewer = None
        self.sim = None
        self.handles = None
        self.kp = None
        self.kd = None
        self.kp_null = None
        self.kd_null = None
        self.cmd_limit = None
        self.states = None
        self._cube_id = None
        self._root_state = None
        self._cube_state = None
        self._dof_state = None
        self._rigid_body_state = None
        self._q = None
        self._qd = None
        self._eef_state = None
        self._eef_lf_state = None
        self._eef_rf_state = None
        self._j_eef = None
        self._mm = None
        self._pos_control = None
        self._effort_control = None
        self._global_indices = None
        self.franka_default_dof_pos = None
        self.franka_dof_lower_limits = None
        self.franka_dof_upper_limits = None
        self._franka_effort_limits = None
        self.envs = None
        self.franka_handles = None
        self.num_dofs = None
        self.num_franka_dofs = None

        if use_gpu_pipeline:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{graphics_device_id}")
                # Warn because currently not supported to use mass_matrix_tensor on GPU
                print(
                    "WARNING: Using GPU pipeline, but mass_matrix_tensor will be on CPU. "
                    "This is currently not supported and will cause an error."
                )
            else:
                print(
                    "WARNING: Using GPU pipeline, but CUDA is not available. "
                    "Falling back to CPU pipeline."
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.cube_size = cube_size
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )
        self.create_sim()
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        self.gym.prepare_sim(self.sim)

        # Create viewer
        if headless:
            self.viewer = None
        else:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            camera_props.horizontal_fov = 60
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
            center_env_index = get_center_env_index(num_envs, int(math.sqrt(num_envs)))
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.envs[center_env_index],
                gymapi.Vec3(6.0, -6.0, 3.0),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )

        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(
                self.envs[0], franka_handle, "panda_hand"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                self.envs[0], franka_handle, "panda_leftfinger_tip"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                self.envs[0], franka_handle, "panda_rightfinger_tip"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                self.envs[0], franka_handle, "panda_grip_site"
            ),
        }

        # OSC Gains
        self.kp = to_torch([150.0] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.0] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)  # Set control gains

        # Set control limits
        self.cmd_limit = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device
        ).unsqueeze(0)

        self.states = {}
        self.initialize_tensor_references()
        self.reset_pose()
        self.reset_cube()

        self._refresh()

    def create_sim(self):
        """Create the simulation."""
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.num_threads
        sim_params.physx.use_gpu = self.use_gpu

        sim_params.use_gpu_pipeline = self.use_gpu_pipeline

        self.sim = self.gym.create_sim(
            self.compute_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, 1.0)

    def _create_ground_plane(self):
        """Create the ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs: int, spacing: float):
        """Create the environments.

        Args:
            num_envs (int): Number of environments to create.
            spacing (float): Spacing between the environments.
        """
        # Load franka asset
        asset_root = "assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
        franka_asset = self.gym.load_asset(
            self.sim, asset_root, franka_asset_file, asset_options
        )

        # Create cube asset
        cube_opts = gymapi.AssetOptions()
        cube_opts.density = 200.0
        cube_asset = self.gym.create_box(self.sim, *([self.cube_size] * 3), cube_opts)
        cube_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Set up the env grid
        self.num_envs = num_envs
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Some common handles for later use
        self.envs = []
        self.franka_handles = []

        franka_dof_stiffness = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 5000.0, 5000.0], dtype=torch.float, device=self.device
        )
        franka_dof_damping = to_torch(
            [0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device
        )

        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        franka_dof_lower_limits = []
        franka_dof_upper_limits = []
        _franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = (
                gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            )
            franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
            franka_dof_props["damping"][i] = franka_dof_damping[i]
            franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            franka_dof_upper_limits.append(franka_dof_props["upper"][i])
            _franka_effort_limits.append(franka_dof_props["effort"][i])

        franka_dof_props["effort"][7] = 200
        franka_dof_props["effort"][8] = 200
        self.franka_dof_lower_limits = to_torch(
            franka_dof_lower_limits, device=self.device
        )
        self.franka_dof_upper_limits = to_torch(
            franka_dof_upper_limits, device=self.device
        )
        self._franka_effort_limits = to_torch(_franka_effort_limits, device=self.device)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        print("Creating %d environments" % self.num_envs)
        num_per_row = int(math.sqrt(self.num_envs))

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add franka
            franka_handle = self.gym.create_actor(
                env, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            self.franka_handles.append(franka_handle)

            # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
            cube_start_pose = gymapi.Transform()
            cube_xy_pos = torch.rand(2) * 0.2
            cube_start_pose.p = gymapi.Vec3(*cube_xy_pos, self.cube_size / 2)
            cube_start_pose.r = gymapi.Quat.from_euler_zyx(
                0, 0, torch.rand(1) * 2 * torch.pi
            )

            # Create cube
            self._cube_id = self.gym.create_actor(
                env, cube_asset, cube_start_pose, "cube", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                env, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color
            )

    def _refresh(self):
        """Refresh the state of the simulation."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self._update_states()

    def initialize_tensor_references(self):
        """Initialize references to tensors that will be updated every step."""
        # Initialize Franka handle
        franka_handle = 0

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Acquire actor root state tensor
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )

        # Get cube state
        self._cube_state = self._root_state[:, self._cube_id, :]

        # Acquire DOF state tensor
        self._dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        ).view(self.num_envs, -1, 2)

        # Get rigid body state
        self._rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)

        # Get values of each state
        self._q = self._dof_state[:, :, 0]
        self._qd = self._dof_state[:, :, 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[
            :, self.handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.handles["rightfinger_tip"], :
        ]

        # Acquire Jacobian tensor
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # Get Jacobian for end effector
        hand_joint_index = self.gym.get_actor_joint_dict(self.envs[0], franka_handle)[
            "panda_hand_joint"
        ]
        self._j_eef = jacobian[:, hand_joint_index, :, :7]

        # Acquire mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices (2: franka and cube)
        self._global_indices = torch.arange(
            self.num_envs * 2, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def _update_states(self):
        """Update the states of the simulation."""
        self.states.update(
            {
                # Franka
                "q": self._q[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "eef_vel": self._eef_state[:, 7:],
                "eef_lf_pos": self._eef_lf_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3],
                # Cubes
                "cube_quat": self._cube_state[:, 3:7],
                "cube_pos": self._cube_state[:, :3],
                "cube_pos_relative": self._cube_state[:, :3] - self._eef_state[:, :3],
            }
        )

    def _compute_osc_torques(self, dpose):
        """Compute the operational space control torques.

        Args:
            dpose (torch.Tensor): Change in pose. The shape is (n, 6), where n is the number of environments and 6 represents the change in position and orientation.
        """
        mm = self._mm
        j_eef = self._j_eef

        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(j_eef, 1, 2)
            @ m_eef
            @ (self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)
        )

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, 7:] *= 0
        u_null = mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye(7, device=self.device).unsqueeze(0)
            - torch.transpose(j_eef, 1, 2) @ j_eef_inv
        ) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0),
        )

        return u

    def pre_physics_step(self, actions):
        """Apply actions to the simulation.

        Args:
            actions (torch.Tensor): Actions to apply to the simulation. The shape is (n, 7), where n is the number of environments and 7 represents the joint torques.
        """
        actions = actions.clone().detach().to(self.device)

        # Split arm and gripper actions
        arm_action = actions[:, :-1]
        gripper_action = actions[:, -1]

        # Scale action to control range
        arm_action = arm_action * self.cmd_limit
        arm_force = self._compute_osc_torques(arm_action)

        # Control gripper
        pos_fingers = torch.zeros((self.num_envs, 2), device=self.device)
        pos_fingers[:, 0] = torch.where(
            gripper_action >= 0.0,
            self.franka_dof_upper_limits[-2].item(),
            self.franka_dof_lower_limits[-2].item(),
        )
        pos_fingers[:, 1] = torch.where(
            gripper_action >= 0.0,
            self.franka_dof_upper_limits[-1].item(),
            self.franka_dof_lower_limits[-1].item(),
        )

        self._pos_control[:, 7:9] = pos_fingers
        self._effort_control[:, :7] = arm_force

        # Apply action to simulation
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )

    def step_physics(self):
        """Step the simulation forward."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def post_physics_step(self):
        """Update the state of the simulation."""
        self._refresh()

    def step_rendering(self):
        """Render the simulation."""
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

    def reset_env(
        self,
        env_ids: Optional[List[int]] = None,
        pose: Literal["default", "middle"] = "default",
        max_cube_radius: float = 0.2,
    ):
        """Reset the pose of the Franka arm and cube.

        Args:
            env_ids (Optional[List[int]], optional): List of environment indices to reset. Defaults to None.
            pose (Literal["default", "middle"], optional): Pose to reset to. Defaults to "default".
            max_cube_radius (float, optional): Maximum radius of the cube from the origin. Defaults to 0.2.
        """
        self.reset_pose(pose, env_ids)
        self.reset_cube(env_ids, max_cube_radius)

    def reset_pose(
        self,
        pose: Literal["default", "middle"] = "default",
        env_ids: Optional[List[int]] = None,
    ):
        """Reset the pose of the Franka arm.

        Args:
            pose (Literal["default", "middle"], optional): Pose to reset to. Defaults to "default".
            env_ids (Optional[List[int]], optional): List of environment indices to reset. Defaults to None.
        """
        # get joint limits and ranges for Franka
        franka_mids = 0.5 * (
            self.franka_dof_upper_limits + self.franka_dof_lower_limits
        )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32)
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.int32)

        if pose == "default":
            position = self.franka_default_dof_pos
        elif pose == "middle":
            position = franka_mids
        else:
            raise ValueError("Invalid pose")

        self._q[env_ids, :] = position
        self._qd[env_ids, :] = torch.zeros_like(self.franka_default_dof_pos)

        self._pos_control[env_ids, :] = position
        self._effort_control[env_ids, :] = torch.zeros_like(position)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def reset_cube(self, env_ids: Optional[List[int]] = None, max_radius: float = 0.2):
        """Reset the pose of the cube.

        Args:
            env_ids (Optional[List[int]], optional): List of environment indices to reset. Defaults to None.
            max_radius (float, optional): Maximum radius of the cube from the origin. Defaults to 0.2.
        """
        # To prevent collisions
        min_radius = self.cube_size * math.sqrt(2)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32)
        else:
            env_ids = torch.tensor(env_ids, dtype=torch.int32)

        # Define the initial pose of the cube
        cube_state = torch.zeros((len(env_ids), 7))
        angles = 2 * math.pi * torch.rand(len(env_ids))
        radii = torch.rand(len(env_ids)) * (max_radius - min_radius) + min_radius

        cube_state[:, 0] = radii * torch.cos(angles)
        cube_state[:, 1] = radii * torch.sin(angles)
        cube_state[:, 2] = self.cube_size / 2

        cube_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids)),
            torch.zeros(len(env_ids)),
            torch.rand(len(env_ids)) * 2 * torch.pi,
        )

        # Set the initial pose of the cube
        self._cube_state[env_ids, :7] = cube_state

        multi_env_ids_cube_int32 = self._global_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cube_int32),
            len(multi_env_ids_cube_int32),
        )

    def destroy(self):
        """Destroy the simulation."""
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def __del__(self):
        self.destroy()
