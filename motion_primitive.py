from typing import List

import torch


class MotionPrimitive:
    def __init__(
        self,
        current_pos: torch.Tensor,
        num_angles: int = 12,
        lengths: List[float] = [0.05, 0.1, 0.15],
        goal_radius: float = 0.01,
    ):
        self.target_pos: torch.Tensor = current_pos.clone()
        self.angles: torch.Tensor = torch.linspace(0, 2 * torch.pi, num_angles)
        self.lengths: torch.Tensor = torch.Tensor(lengths)
        # self.current_angle_indices: torch.Tensor = torch.zeros(current_pos.shape[0], dtype=torch.long)
        # self.current_length_indices: torch.Tensor = torch.zeros(current_pos.shape[0], dtype=torch.long)
        self.goal_radius: float = goal_radius

    # Check if current position is within goal radius
    def _is_goal(self, current_pos: torch.Tensor) -> torch.Tensor:
        return torch.norm(current_pos - self.target_pos, dim=1) < self.goal_radius

    def _update_target(
        self,
        current_pos: torch.Tensor,
        cube_pos_relative: torch.Tensor,
        epsilon: float = 0.5,
    ) -> bool:
        is_goal = self._is_goal(current_pos)
        num_goals = is_goal.sum()

        if num_goals == 0:
            return False
        else:
            relative_pos = cube_pos_relative[is_goal]
            normalized_relative_pos = relative_pos / torch.norm(
                relative_pos, dim=1, keepdim=True
            )

            # Update target position of the primitives that reached the goal
            if_random = torch.rand(num_goals) < epsilon

            random_angle_indices = torch.randint(0, len(self.angles), (num_goals,))

            expanded_x_angles = torch.cos(self.angles).unsqueeze(0).repeat(num_goals, 1)
            expanded_y_angles = torch.sin(self.angles).unsqueeze(0).repeat(num_goals, 1)

            expanded_normalized_relative_pos = normalized_relative_pos.unsqueeze(
                1
            ).repeat(1, len(self.angles), 1)

            dx = expanded_x_angles - expanded_normalized_relative_pos[..., 0]
            dy = expanded_y_angles - expanded_normalized_relative_pos[..., 1]

            dist_sq = dx**2 + dy**2

            closest_angle_indices = torch.argmin(dist_sq, dim=1)

            angle_indices = torch.where(
                if_random, random_angle_indices, closest_angle_indices
            )
            length_indices = torch.randint(0, len(self.lengths), (num_goals,))

            # self.current_angle_indices[is_goal] = angle_indices
            # self.current_length_indices[is_goal] = length_indices

            angles = self.angles[angle_indices]
            lengths = self.lengths[length_indices]

            self.target_pos[is_goal] += lengths.unsqueeze(-1) * torch.stack(
                [torch.cos(angles), torch.sin(angles)], dim=1
            )
            return True

    def get_relative_goal(
        self, current_pos: torch.Tensor, cube_pos_relative: torch.Tensor
    ) -> torch.Tensor:
        self._update_target(current_pos, cube_pos_relative)
        return self.target_pos - current_pos
