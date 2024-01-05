import torch


def wrap_angle(angle):
    """Wrap an angle to the range [-pi, pi]"""
    return angle - 2 * torch.pi * torch.floor((angle + torch.pi) / (2 * torch.pi))
