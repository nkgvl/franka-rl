import os
from typing import Dict, List

import torch


def primitive_to_dxy(primitive: torch.Tensor) -> torch.Tensor:
    """Translate from primitive to difference of coordinates.

    Args:
        primitive (torch.Tensor): Primitive of shape (2, ) where 2 represents the angle and length.

    Returns:
        torch.Tensor: Difference of coordinates of shape (2, ) where 2 represents the x and y coordinates.
    """
    return torch.tensor(
        [
            primitive[1] * torch.cos(primitive[0]),
            primitive[1] * torch.sin(primitive[0]),
        ]
    )


class MotionPrimitiveController:
    """Motion primitive controller for the end effectors."""

    def __init__(
        self,
        current_positions: torch.Tensor,
        goal_radius: float = 0.01,
        moving_threshold: float = 0.001,
        height: float = 0.025,
    ) -> None:
        """Initialize the status of the motion primitives.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.
            goal_radius (float, optional): Radius of the goal. Defaults to 0.01.
            moving_threshold (float, optional): Threshold for determining if the end effector is moving. Defaults to 0.001.
            height (float, optional): Height of the end effectors. Defaults to 0.025.
        """
        num_envs = current_positions.shape[0]
        self.num_envs = num_envs
        self.height = height

        self._current_positions = None
        self.current_positions = current_positions

        self._current_primitives = None
        self.current_primitives = torch.zeros((num_envs, 3))

        self._current_targets = None
        self.current_targets = torch.zeros_like(current_positions)
        self.current_targets[:, 2] = height

        self._dones = None
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

        self.trajectories = [
            {"states": [], "actions": [], "dones": []} for _ in range(self.num_envs)
        ]

    @property
    def current_positions(self):
        """Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates."""
        return self._current_positions

    @current_positions.setter
    def current_positions(self, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("current_positions must be a torch.Tensor")
        if value.shape != (self.num_envs, 3):
            raise ValueError("current_positions must have shape (num_envs, 3)")
        self._current_positions = value

    @property
    def current_primitives(self):
        """Current primitives of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the angle, length and height."""
        return self._current_primitives

    @current_primitives.setter
    def current_primitives(self, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("current_primitives must be a torch.Tensor")
        if value.shape != (self.num_envs, 3):
            raise ValueError("current_primitives must have shape (num_envs, 3)")
        self._current_primitives = value

    @property
    def current_targets(self):
        """Current targets of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates."""
        return self._current_targets

    @current_targets.setter
    def current_targets(self, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("current_targets must be a torch.Tensor")
        if value.shape != (self.num_envs, 3):
            raise ValueError("current_targets must have shape (num_envs, 3)")
        self._current_targets = value

    @property
    def dones(self):
        """Boolean tensor of shape (n, ) where n is the number of environments."""
        return self._dones

    @dones.setter
    def dones(self, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("dones must be a torch.Tensor")
        if value.shape != (self.num_envs,):
            raise ValueError("dones must have shape (num_envs,)")
        self._dones = value

    def is_within_eef_bounds(self, current_positions: torch.Tensor) -> torch.Tensor:
        """Check if the end effectors are within the bounds.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape (n, ) where n is the number of environments. Returns True if the end effector is within bounds, False otherwise.
        """
        within_radius = (
            torch.norm(current_positions[:, :2] - torch.Tensor([0.5, 0]), dim=1) <= 0.75
        )
        within_x_range = current_positions[:, 0] > -0.25
        correct_height = torch.abs(current_positions[:, 2] - self.height) <= 0.01

        return torch.logical_and(
            within_radius, torch.logical_and(within_x_range, correct_height)
        )

    def update(
        self,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
        cube_pos_relative: torch.Tensor,
        states_dict: Dict[str, torch.Tensor],
        epsilon: float = 0.5,
        force: bool = False,
    ) -> List[int]:
        """Update the status of the motion primitives.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.
            current_velocities (torch.Tensor): Current velocities of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z velocities.
            cube_pos_relative (torch.Tensor): Relative position of the cube. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.
            force (bool, optional): Force to update the motion primitives. Defaults to False.

        Returns:
            List[int]: List of indices of the environments that finished the episode.
        """
        self._update_targets(
            current_positions,
            current_velocities,
            cube_pos_relative,
            states_dict,
            epsilon,
            force,
        )

        return self.dones.nonzero(as_tuple=True)[0].detach().cpu().numpy().tolist()

    def _is_goal(self, current_positions: torch.Tensor) -> torch.Tensor:
        """Check if the end effectors reached the goal.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape (n, ) where n is the number of environments.
        """
        return (
            torch.norm(current_positions - self.current_targets, dim=1)
            < self.goal_radius
        )

    def _is_moving(self, current_velocities: torch.Tensor) -> torch.Tensor:
        """Check if the end effectors are moving.

        Args:
            current_vel (torch.Tensor): Current velocities of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z velocities.

        Returns:
            torch.Tensor: Boolean tensor of shape (n, ) where n is the number of environments.
        """
        return torch.norm(current_velocities, dim=1) > self.moving_threshold

    def _update_targets(
        self,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
        cube_pos_relative: torch.Tensor,
        states_dict: Dict[str, torch.Tensor],
        epsilon: float,
        force: bool = False,
    ) -> None:
        """Update the target positions of the motion primitives.

        Args:
            current_positions (torch.Tensor): Current positions of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.
            current_velocities (torch.Tensor): Current velocities of the end effectors. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z velocities.
            cube_pos_relative (torch.Tensor): Relative position of the cube. The shape is (n, 3), where n is the number of environments and 3 represents the x, y and z coordinates.
            states_dict (Dict[str, torch.Tensor]): Dictionary of states.
            epsilon (float): Probability of choosing a random primitive.
            force (bool, optional): Force to update the motion primitives. Defaults to False.

        Returns:
            torch.Tensor: List of indices of the environments that reached the goal.
        """
        if force:
            indices = [i for i in range(len(current_positions))]
        else:
            is_goal = self._is_goal(current_positions)
            is_out_of_bounds = ~self.is_within_eef_bounds(current_positions)
            self.dones[is_out_of_bounds] = True
            indices = (is_goal | is_out_of_bounds).nonzero(as_tuple=True)[0]
        for i in indices:
            self._update_primitive(i, cube_pos_relative[i], states_dict, epsilon)
            self._update_target(i, current_positions[i])

    def _update_target(
        self,
        index: int,
        current_position: torch.Tensor,
    ) -> bool:
        """Update the target position of the motion primitive.

        Args:
            index (int): Index of the environment.
            current_position (torch.Tensor): Current position of the end effector. The shape is (3, ) where 3 represents the x, y and z coordinates.
        """
        self.current_targets[index, :2] = current_position[:2] + primitive_to_dxy(
            self.current_primitives[index]
        )

    def _update_primitive(
        self,
        index: int,
        cube_pos_relative: torch.Tensor,
        states_dict: Dict[str, torch.Tensor],
        epsilon: float,
    ) -> None:
        """Update the motion primitive. It is called when the end effector reaches the goal or goes out of bounds.

        Args:
            index (int): Index of the environment.
            cube_pos_relative (torch.Tensor): Relative position of the cube. The shape is (3, ) where 3 represents the x, y and z coordinates.s
            states_dict (Dict[str, torch.Tensor]): Dictionary of states.
            epsilon (float): Probability of choosing a random primitive.
        """
        if torch.rand(1) < epsilon:
            angle_index = torch.randint(0, len(self.primitive_angles), (1,))
            length_index = torch.randint(0, len(self.primitive_lengths), (1,))
            self.current_primitives[index] = torch.tensor(
                [
                    self.primitive_angles[angle_index],
                    self.primitive_lengths[length_index],
                    self.height,
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
                    self.height,
                ]
            )

        # Save trajectory
        self.trajectories[index]["states"].append(
            {key: states_dict[key][index] for key in states_dict.keys()}
        )
        self.trajectories[index]["actions"].append(self.current_primitives[index])
        self.trajectories[index]["dones"].append(self.dones[index])

    def save_trajectories(self, filename: str, directory: str = "trajectories") -> None:
        """Save the trajectories to a file.

        Args:
            filename (str): Filename.
            directory (str, optional): Directory. Defaults to "trajectories".
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.trajectories, os.path.join(directory, filename))

    def clear_trajectories(self) -> None:
        """Clear the trajectories."""
        self.trajectories = [
            {"states": [], "actions": [], "dones": []} for _ in range(self.num_envs)
        ]
