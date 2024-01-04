from typing import List

import torch


class MotionPrimitiveController:
    def __init__(
        self,
        current_positions: torch.Tensor,
        goal_radius: float = 0.01,
        moving_threshold: float = 0.001,
    ) -> None:
        """Initialize the status of the motion primitives.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.
            goal_radius (float, optional): Radius of the goal. Defaults to 0.01.
            moving_threshold (float, optional): Threshold for determining if the end effector is moving. Defaults to 0.001.
        """
        num_envs = current_positions.shape[0]
        self.current_positions = current_positions
        self.current_primitives = torch.zeros(
            (num_envs, 2), dtype=torch.long
        )  # 2: (angle, length)
        self.current_targets = torch.zeros_like(current_positions)
        self.dones = torch.zeros(current_positions.shape[0], dtype=torch.bool)

        self.num_primitive_angles = 12
        self.primitive_angles = torch.tensor(
            [
                i * 2 * torch.pi / self.num_primitive_angles
                for i in range(self.num_primitive_angles)
            ]
        )
        self.primitive_lengths = torch.tensor([0.05, 0.1, 0.15])

        self.goal_radius = goal_radius
        self.moving_threshold = moving_threshold

        self.trajectories = [[] for _ in range(num_envs)]  # Added this line

    def update(
        self,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
        cube_pos_relative: torch.Tensor,
    ) -> List[int]:
        """Update the status of the motion primitives.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.
            cube_pos_relative (torch.Tensor): Relative position of the cube. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.

        Returns:
            List[int]: List of indices of the environments that reached the goal.
        """
        self._update_target(
            current_positions, current_velocities, cube_pos_relative, 0.5
        )

    def _is_goal(self, current_positions: torch.Tensor) -> torch.Tensor:
        """Check if the end effectors reached the goal.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape (n, ) where n is the number of environments.
        """
        return (
            torch.norm(current_positions - self.current_targets, dim=1)
            < self.goal_radius
        )

    def _is_moving(self, current_vel: torch.Tensor) -> torch.Tensor:
        """Check if the end effectors are moving.

        Args:
            current_vel (torch.Tensor): Current velocities of the end effectors. The shape is (n, 2), where n is the number of environments and 2 represents the x and y coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape (n, ) where n is the number of environments.
        """
        return (
            torch.norm(current_vel - self.current_targets, dim=1)
            > self.moving_threshold
        )

    def _update_target(
        self,
        current_positions: torch.Tensor,
        current_vel: torch.Tensor,
        cube_pos_relative: torch.Tensor,
        epsilon: float,
    ) -> bool:
        is_goal = self._is_goal(current_positions)
        is_not_moving = ~self._is_moving(current_vel)
        self.dones[is_goal | is_not_moving] = 1
        indices = is_goal.nonzero(as_tuple=True)[0]
        for i in indices:
            self._update_primitive(i, cube_pos_relative[i], epsilon)

    def _update_primitive(
        self, index: int, cube_pos_relative: torch.Tensor, epsilon: float
    ) -> None:
        if torch.rand(1) < epsilon:
            angle_index = torch.randint(0, len(self.primitive_angles), (1,))
            length_index = torch.randint(0, len(self.primitive_lengths), (1,))
            self.current_primitives[index] = torch.tensor(
                [
                    self.primitive_angles[angle_index],
                    self.primitive_lengths[length_index],
                ]
            )
        else:
            angle_to_cube = torch.atan2(cube_pos_relative[1], cube_pos_relative[0])
            angle_diffs = torch.abs(self.primitive_angles - angle_to_cube)
            closest_angle_index = torch.argmin(angle_diffs)
            length_index = torch.randint(0, len(self.primitive_lengths), (1,))
            self.current_primitives[index] = torch.tensor(
                [
                    self.primitive_angles[closest_angle_index],
                    self.primitive_lengths[length_index],
                ]
            )
        self.trajectories[index].append(
            (
                self.current_positions[index].clone(),  # state
                self.current_primitives[index].clone(),  # action
                self.dones[index].clone(),  # done
            )
        )
