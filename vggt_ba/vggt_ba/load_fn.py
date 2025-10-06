
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of vggt main repo

from typing import List
from fractions import Fraction

import torch
from PIL import Image
import numpy as np
from torchvision import transforms as TF


def load_images(image_path_list):
    non_preprocessed_imgs = [
        Image.open(image_path)
        for image_path in image_path_list
    ]
    return non_preprocessed_imgs


def preprocess_images(image_list, mode="crop"):
    # Check for empty list
    if len(image_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img in image_list:
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images



def preprocess_images_square(pil_images, target_size=1024):
    """
    Preprocess a list of PIL Images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        pil_images (list): List of PIL.Image objects
        target_size (int, optional): Target size for both width and height. Defaults to 1024.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 6) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    if len(pil_images) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []
    to_tensor = TF.to_tensor

    for img in pil_images:
        # Ensure image is a PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError("All elements in pil_images must be PIL.Image objects")

        # Handle transparency
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Compute scale factor
        scale = target_size / max_dim

        # Final coordinates of original image in target image space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Pad image to square
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack into batch
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    return images, original_coords


def check_aspect_ratios(image_paths: List[str]) -> bool:
    """
    Check if all images have the same aspect ratio.

    Parameters:
        image_paths (List[str]): List of paths to image files.

    Returns:
        bool: True if all images have the same aspect ratio, False otherwise.
    """
    if not image_paths:
        raise ValueError("No image paths provided.")

    aspect_ratios = []

    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
            aspect_ratio = Fraction(width, height).limit_denominator()
            aspect_ratios.append(aspect_ratio)

    first_ratio = aspect_ratios[0]
    all_same = all(ratio == first_ratio for ratio in aspect_ratios)

    return all_same


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    return image.crop((left, top, right, bottom))


def crop_images_to_square(images):
    return [center_crop_square(img) for img in images]
