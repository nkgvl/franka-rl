from typing import Optional

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *


class PositionController:
    """Position controller for the end effector."""

    def __init__(self, env, height: float = 0.025, gain: float = 10.0):
        """Initialize the position controller.

        Args:
            env (FrankaPandaEnv): Environment.
            height (float, optional): Height of the end effector. Defaults to 0.025.
            gain (float, optional): Gain of the position controller. Defaults to 10.0.
        """
        self.env = env
        self.target_positions: Optional[torch.Tensor] = None
        self.height: float = height
        self.gain: float = gain

        self._move_to_default_pos()
        self._set_default_dof_pos()

    def _move_to_default_pos(self):
        for _ in range(1000):
            dpose_xy = torch.zeros((self.env.num_envs, 2))
            dpose_xy[:, :2] = torch.Tensor([0, 0]) - self.env.states["eef_pos"][:, :2]
            self.set_force(dpose_xy)
            self.env.step_physics()
            self.env.post_physics_step()

    def _set_default_dof_pos(self):
        pose = (
            gymtorch.wrap_tensor(self.env.gym.acquire_dof_state_tensor(self.env.sim))
            .view(self.env.num_envs, -1, 2)[:, :, 0]
            .mean(0)
            .clone()
        )
        self.env.franka_default_dof_pos = pose

    def set_force(self, dpose_xy):
        """Set the force to the end effector.

        Args:
            dpose_xy (torch.Tensor): Desired change of the end effector position in the x and y directions. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.
        """
        dpose = torch.zeros((self.env.num_envs, 7))
        dpose[:, :3] = torch.Tensor([0, 0, self.height]) - self.env.states["eef_pos"]
        dpose[:, :2] = dpose_xy

        target_quat = torch.Tensor([1, 0, 0, 0])
        current_quat = self.env.states["eef_quat"]
        diff_quat = quat_mul(
            target_quat.unsqueeze(0).expand_as(current_quat),
            quat_conjugate(current_quat),
        )
        euler_angles = get_euler_xyz(diff_quat)
        gain_rot = 2.0
        dpose[:, 3], dpose[:, 4], dpose[:, 5] = (
            gain_rot * normalize_angle(euler_angles[0]),
            gain_rot * normalize_angle(euler_angles[1]),
            gain_rot * normalize_angle(euler_angles[2]),
        )
        dpose[:, -1] = -1

        self.env.pre_physics_step(dpose * self.gain)

    def set_target(self, target_positions: torch.Tensor):
        """Set the target position.

        Args:
            target_positions (torch.Tensor): Target positions of the end effectors. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.
        """
        self.target_positions = target_positions

    def step(self):
        """Step the simulation."""
        if self.target_positions is None:
            raise ValueError("Target position is not set.")
        dpose_xy = self.target_positions - self.env.states["eef_pos"][:, :2]
        self.set_force(dpose_xy)
        self.env.step_physics()
        self.env.post_physics_step()
        self.env.step_rendering()
