import imageio
import torch
import numpy as np

MEANS = [224.0111, 235.0948, 226.7944]
STDS = [69.3141, 54.4987, 63.4723]


def load_gif(path: str) -> torch.Tensor:
    """
    Loads a .gif file and returns the frames as a tensor.

    Args:
    - path (str): The path to the .gif file.

    Returns:
    - torch.Tensor: The tensor containing the frames of the .gif file. Shape: (B, S, C, H, W)
    """
    gif = imageio.get_reader(path)
    frames = np.array([frame for frame in gif])
    frames = np.transpose(frames, (0, 3, 1, 2))
    tensor_frames = torch.tensor(frames, dtype=torch.float32)  # Shape: (S, C, H, W)
    batched_tensor = tensor_frames.unsqueeze(0)  # Add batch dimension (B=1)
    return batched_tensor


def crop_to_field_of_view(image: torch.Tensor, upper_left: int = 73,
                          lower_right: int = 73 + 461, upper_right: int = 101,
                          lower_left: int = 101 + 495) -> torch.Tensor:
    """
    Crops the batched images tensor to the field of view.

    Args:
    - image (torch.Tensor): The image tensor to crop.

    Returns:
    - torch.Tensor: The cropped image tensor.
    """
    return image[..., upper_left:lower_right, upper_right:lower_left]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalizes batched image tensor using the precomputed mean and std values.

    Args:
    - image (torch.Tensor): The image tensor to normalize.

    Returns:
    - torch.Tensor: The normalized image tensor.
    """
    mean = torch.tensor(MEANS).view(3, 1, 1)
    std = torch.tensor(STDS).view(3, 1, 1)
    return (image - mean) / std


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Unnormalizes batched image tensor using the precomputed mean and std values.

    Args:
    - image (torch.Tensor): The image tensor to unnormalize.

    Returns:
    - torch.Tensor: The unnormalized image tensor.
    """
    mean = torch.tensor(MEANS).view(3, 1, 1)
    std = torch.tensor(STDS).view(3, 1, 1)
    return image * std + mean


def transform_gif_to_tensor(gif_path: str) -> torch.Tensor:
    """
    Transforms a .gif file to a normalized, cropped tensor.

    Args:
    - gif_path (str): The path to the .gif file.

    Returns:
    - torch.Tensor: The tensor containing the frames of the .gif file. Shape: (B, S, C, H, W)
    """
    frames = load_gif(gif_path)
    frames = crop_to_field_of_view(frames)
    frames = normalize_image(frames)
    return frames









