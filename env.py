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
from typing import Literal

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *


class FrankaPandaEnv:
    def __init__(
        self,
        num_envs: int = 36,
        physics_engine: int = gymapi.SIM_PHYSX,
        use_gpu: bool = True,
        use_gpu_pipeline: bool = True,
        num_threads: int = 1,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
    ):
        self.gym = gymapi.acquire_gym()
        # Set device information
        # self.device = torch.device(
        #     f"cuda:{compute_device_id}" if torch.cuda.is_available() else "cpu"
        # )
        self.device = torch.device("cpu")

        self.physics_engine = physics_engine
        self.use_gpu = use_gpu
        self.use_gpu_pipeline = use_gpu_pipeline
        self.num_threads = num_threads
        self.compute_device_id = compute_device_id
        self.graphics_device_id = graphics_device_id

        self.create_sim()

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

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

        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

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
        self._update_states()

        self._refresh()

    def create_sim(self):
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        if self.physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 15
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
        elif self.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.num_threads
            sim_params.physx.use_gpu = self.use_gpu

        sim_params.use_gpu_pipeline = self.use_gpu_pipeline

        self.sim = self.gym.create_sim(
            self.compute_device_id,
            self.graphics_device_id,
            self.physics_engine,
            sim_params,
        )
        self._create_ground_plane()
        self._create_envs(36, 1.0)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs: int, spacing: float):
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
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
                franka_dof_props["damping"][i] = franka_dof_damping[i]
            else:
                franka_dof_props["stiffness"][i] = 7000.0
                franka_dof_props["damping"][i] = 50.0
            franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            _franka_effort_limits.append(franka_dof_props['effort'][i])

        franka_dof_props["effort"][7] = 200
        franka_dof_props["effort"][8] = 200
        self.franka_dof_lower_limits = to_torch(franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(franka_dof_upper_limits, device=self.device)
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
                env, franka_asset, franka_start_pose, "franka", i, 2
            )
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            self.franka_handles.append(franka_handle)

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self._update_states()

    def _update_states(self):
        _dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        ).view(self.num_envs, -1, 2)
        _rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)
        _q = _dof_state[..., 0]
        _eef_state = _rigid_body_state[:, self.handles["grip_site"], :]
        _eef_lf_state = _rigid_body_state[:, self.handles["leftfinger_tip"], :]
        _eef_rf_state = _rigid_body_state[:, self.handles["rightfinger_tip"], :]
        self.states.update(
            {
                # Franka
                "q": _q[:, :],
                "q_gripper": _q[:, -2:],
                "eef_pos": _eef_state[:, :3],
                "eef_quat": _eef_state[:, 3:7],
                "eef_vel": _eef_state[:, 7:],
                "eef_lf_pos": _eef_lf_state[:, :3],
                "eef_rf_pos": _eef_rf_state[:, :3],
            }
        )

    def _compute_osc_torques(self, dpose):
        dof_state_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        ).view(self.num_envs, self.num_franka_dofs, 2)
        mm = gymtorch.wrap_tensor(
            self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        )
        jacobian = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "franka")
        )
        hand_joint_index = self.gym.get_actor_joint_dict(
            self.envs[0], self.franka_handles[0]
        )["panda_hand_joint"]

        mm = mm[:, :7, :7]
        j_eef = jacobian[:, hand_joint_index, :, :7]

        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = dof_state_tensor[:, :7, 0], dof_state_tensor[:, :7, 1]
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
        franka_dof_props = self.gym.get_actor_dof_properties(
            self.envs[0], self.franka_handles[0]
        )
        franka_effort_limits = to_torch(franka_dof_props["effort"], device=self.device)

        u = tensor_clamp(
            u.squeeze(-1),
            -franka_effort_limits[:7].unsqueeze(0),
            franka_effort_limits[:7].unsqueeze(0),
        )

        return u

    def pre_physics_step(self, actions):
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

        control = torch.cat([arm_force, pos_fingers], dim=1)

        # Apply action to simulation
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(control)
        )

    def step_physics(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def step_rendering(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    def reset_pose(self, pose: Literal["default", "middle"] = "default"):
        # get joint limits and ranges for Franka
        franka_dof_props = self.gym.get_actor_dof_properties(
            self.envs[0], self.franka_handles[0]
        )
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_upper_limits - franka_lower_limits
        franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)

        for i in range(self.num_envs):
            # Set updated stiffness and damping properties
            self.gym.set_actor_dof_properties(
                self.envs[i], self.franka_handles[i], franka_dof_props
            )

            # Set ranka pose so that each joint is in the middle of its actuation range
            franka_dof_states = self.gym.get_actor_dof_states(
                self.envs[i], self.franka_handles[i], gymapi.STATE_NONE
            )
            if pose == "middle":
                for j in range(self.num_franka_dofs):
                    franka_dof_states["pos"][j] = franka_mids[j]
            elif pose == "default":
                for j in range(self.num_franka_dofs):
                    franka_dof_states["pos"][j] = self.franka_default_dof_pos[j]
            else:
                raise ValueError("Invalid pose argument")
            self.gym.set_actor_dof_states(
                self.envs[i],
                self.franka_handles[i],
                franka_dof_states,
                gymapi.STATE_POS,
            )

    def destroy(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def __del__(self):
        self.destroy()
