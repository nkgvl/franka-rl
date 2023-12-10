from typing import List, Tuple

import torch


class MotionPrimitive:
    def __init__(
        self,
        num_envs: int,
        num_angles: int = 12,
        num_lengths: int = 3,
        time_step: float = 1 / 60.0,
        height: float = 0.03,
    ):
        self.num_envs: int = num_envs
        self.angles: torch.Tensor = torch.linspace(0, 2 * torch.pi, num_angles)
        self.lengths: torch.Tensor = torch.linspace(0.2, 0.6, num_lengths)
        self.height: float = height
        self.time_step: float = time_step

        # Initialize state for each environment
        self.remaining_times: torch.Tensor = torch.zeros(num_envs)
        self.current_angle_indices: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long
        )
        self.current_length_indices: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long
        )
        self.selected_primitives: List[List[Tuple[int, int]]] = [
            [] for _ in range(num_envs)
        ]

    def sample_actions(self) -> torch.Tensor:
        # Update remaining time and sample new primitives if needed
        self.remaining_times -= self.time_step
        new_samples_mask: torch.Tensor = self.remaining_times <= 0

        if new_samples_mask.any():
            self.current_angle_indices[new_samples_mask] = torch.randint(
                0, len(self.angles), (new_samples_mask.sum(),)
            )
            self.current_length_indices[new_samples_mask] = torch.randint(
                0, len(self.lengths), (new_samples_mask.sum(),)
            )
            self.remaining_times[new_samples_mask] = self.lengths[
                self.current_length_indices[new_samples_mask]
            ]

            # Log selected primitives
            for env_idx in torch.where(new_samples_mask)[0]:
                self.selected_primitives[env_idx].append(
                    (
                        self.current_angle_indices[env_idx].item(),
                        self.current_length_indices[env_idx].item(),
                    )
                )

        # Generate actions
        angle: torch.Tensor = self.angles[self.current_angle_indices]
        actions: torch.Tensor = torch.zeros((self.num_envs, 2))
        actions[:, 0] = torch.cos(angle)
        actions[:, 1] = torch.sin(angle)

        return actions
