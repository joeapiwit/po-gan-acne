# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import re
from typing import List
from pathlib import Path


import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torchvision.transforms import functional as TF


import legacy
from projector import project


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def load_images_from_folder(folder_path, image_size, device):
    """
    Loads images from a given folder, preprocesses them to match the expected input format
    for the generator, and returns them as a list of PyTorch tensors.

    Parameters:
    - folder_path: Path to the folder containing images.
    - image_size: The target size to which images will be resized (expects a tuple, e.g., (1024, 1024)).
    - device: The device on which to load the tensors ('cuda' or 'cpu').

    Returns:
    - A list of preprocessed images as PyTorch tensors.
    """
    folder = Path(folder_path)
    images = []

    for img_path in folder.glob('*'):
        # Ensure only valid image files are processed
        if not img_path.is_file() or not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            continue

        # Load and preprocess the image
        image = PIL.Image.open(img_path).convert('RGB')
        image = TF.resize(image, image_size)
        image_tensor = TF.to_tensor(image).to(device) * 255  # Scale to [0, 255] if necessary
        images.append(image_tensor)

    return images


def project_images_to_w(target_images, G, device, num_steps=1000, w_avg_samples=10000, initial_learning_rate=0.1, truncation_psi=1.0, verbose=False):

    """
    Project given images to the latent W space of a pretrained StyleGAN model.

    Parameters:
    - target_images: A list of PIL.Image objects or a single PIL.Image object.
    - G: The pretrained StyleGAN generator model.
    - device: The device to run the projection on ('cuda' or 'cpu').
    - num_steps: Number of optimization steps for the projection.
    - w_avg_samples: Number of samples to compute the average W vector.
    - initial_learning_rate: The initial learning rate for the optimizer.
    - verbose: If True, print progress messages.

    Returns:
    - A list of projected W vectors corresponding to the input images.
    """

    # Ensure target_images is a list even if a single image is provided
    if not isinstance(target_images, list):
        target_images = [target_images]

    # Function to convert PIL image to tensor matching G's input format
    def prepare_image(image, device):
        """
        Prepares an image for projection by ensuring it is a tensor and moving it to the specified device.

        Parameters:
        - image: A PIL.Image object or a torch.Tensor.
        - device: The device to move the tensor to ('cuda' or 'cpu').

        Returns:
        - A torch.Tensor of the image.
        """
        if not isinstance(image, torch.Tensor):
            # If the input is not a tensor, convert it to a tensor
            image = TF.to_tensor(image).to(device) * 255  # Scale to [0, 255]
        else:
            # If it's already a tensor, just ensure it's on the correct device
            image = image.to(device)
        image = TF.resize(image, [G.img_resolution, G.img_resolution])
        return image.unsqueeze(0)  # Ensure it has a batch dimension
    
    projected_ws = []

    # Compute the average W vector (w_avg) for truncation, if not already provided
    w_avg = G.mapping.w_avg

    for target_image in target_images:
        target_tensor = prepare_image(target_image, device)  
        # Run the projection
        projected_w_steps = project(
            G,
            target=target_tensor.squeeze(0),  # Remove batch dimension for project function compatibility
            num_steps=num_steps,
            device=device,
            verbose=verbose,
            w_avg_samples=w_avg_samples,
            initial_learning_rate=initial_learning_rate
        )
        # Use the last W vector as the result and apply truncation
        final_w = projected_w_steps[-1]
        # Normalize the W vector using truncation psi and w_avg
        truncated_w = w_avg + (final_w - w_avg) * truncation_psi
        projected_ws.append(truncated_w)

    return projected_ws

# Helper function to save grid canvas
def save_grid_canvas(grid_canvas, outdir, filename):
    grid_image_path = os.path.join(outdir, filename)
    grid_canvas.save(grid_image_path)
    print(f'Saved grid image: {grid_image_path}')
    
# Function to create and save grid images
def generate_and_save_grids(outdir, row_images, col_images, style_indices, include_originals, W, H):
    # Define the canvas size
    num_rows = len(row_images) + (1 if include_originals else 0)
    num_cols = len(col_images) + (1 if include_originals else 0)
    canvas_width = W * num_cols
    canvas_height = H * num_rows
    
    # Create a new canvas
    grid_canvas = PIL.Image.new('RGB', (canvas_width, canvas_height), 'black')
    
    # Function to paste an image into the canvas at the specified position
    def paste_image(image_input, position):
        if isinstance(image_input, str):
            # Assuming image_input is a path to an image file
            if os.path.exists(image_input):
                img = PIL.Image.open(image_input)
                grid_canvas.paste(img, position)
        elif isinstance(image_input, PIL.Image.Image):
            # Assuming image_input is already a PIL Image object
            grid_canvas.paste(image_input, position)
        else:
            raise ValueError("Unsupported image input type.")

    
    # Paste the original row and column images if required
    if include_originals:
        for row_idx, row_image_tensor in enumerate(row_images):
            row_image = TF.to_pil_image(row_image_tensor.cpu() / 255.0)
            paste_image(row_image, (0, (row_idx + 1) * H))  # Offset for column originals
        
        for col_idx, col_image_tensor in enumerate(col_images):
            col_image = TF.to_pil_image(col_image_tensor.cpu() / 255.0)
            paste_image(col_image, ((col_idx + 1) * W, 0))  # Offset for row originals

    # Use the saved images for mixing
    for row_idx in range(len(row_images)):
        for col_idx in range(len(col_images)):
            # Constructing the filename based on the saved naming convention
            image_filename = os.path.join(outdir, f'row-{row_idx}-col-{col_idx}-style-layers-{min(style_indices)}-{max(style_indices)}.png')
            paste_position = ((col_idx + 1) * W, (row_idx + 1) * H)  # Offset to account for originals
            paste_image(image_filename, paste_position)

    # Save the comprehensive grid image
    grid_filename = 'comprehensive-grid.png'
    save_grid_canvas(grid_canvas, outdir, grid_filename)




#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows-folder', 'rows_folder', type=str, help='Folder path with images for rows', required=True)
@click.option('--cols-folder', 'cols_folder', type=str, help='Folder path with images for columns', required=True)
@click.option('--styles', 'col_styles', type=str, help='Style layer range (e.g., 0-6)', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--include-originals', is_flag=True, default=True, help='Include original images in the grid')
def generate_style_mix(
    network_pkl: str,
    rows_folder: str,
    cols_folder: str,
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    include_originals: bool,
):
    """Generate style-mixed images using pretrained network pickle, based on input images."""
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Define W and H based on the generator's image resolution
    W, H = G.img_resolution, G.img_resolution
    image_size = G.img_resolution  # Assuming G.img_resolution provides the model's expected image resolution

    # Create directories for saving individual images and grids if they don't exist
    # os.makedirs(os.path.join(outdir, 'individual'), exist_ok=True)
    # os.makedirs(os.path.join(outdir, 'grids'), exist_ok=True)
    
    os.makedirs(outdir, exist_ok=True)

    # Functions to load and preprocess images, and to project images to W space
    # (Implementations needed here)

    print('Loading and projecting row images...', 'size:', image_size)
    # Update the function call with the correct arguments
    row_images = load_images_from_folder(rows_folder, (W, H), device)
    row_w = project_images_to_w(row_images, G, device)

    print('Loading and projecting column images...', 'size:', image_size)
    # Similarly, for column images
    col_images = load_images_from_folder(cols_folder, (W, H), device)
    col_w = project_images_to_w(col_images, G, device)
    
    # Parse col_styles to get the list of style indices
    start_style, end_style = map(int, col_styles.split('-'))
    style_indices = range(start_style, end_style + 1)

    # Debug print to check style_indices
    print("Style indices:", list(style_indices))
    
    # Generating and mixing images from W vectors
    print('Generating style-mixed images...')
    for i, row_w_vec in enumerate(row_w):
        for j, col_w_vec in enumerate(col_w):
            # Create a mixed W vector based on the row vector
            mixed_w = row_w_vec.clone()

            # Iterate through each style layer you want to mix
            for idx in style_indices:
                # Correctly mix styles without exceeding tensor dimensions
                print('check shape, mixed_w:', mixed_w.shape, 'col_w_vec:', col_w_vec.shape, 'idx:', idx)
                mixed_w[idx, :] = col_w_vec[idx, :]  # Mix style from column vector into the row vector

            # Generate the mixed image using the mixed W vector
            # Ensure mixed_w is properly reshaped or batched if your generator expects a specific input shape
            mixed_image = G.synthesis(mixed_w.unsqueeze(0), noise_mode=noise_mode)  # Add batch dimension if needed
            mixed_image = (mixed_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]

            # Save the mixed image
            image_filename = f'row-{i}-col-{j}-style-layers-{start_style}-{end_style}.png'
            PIL.Image.fromarray(mixed_image, 'RGB').save(os.path.join(outdir, image_filename))
            print(f'Saved: {image_filename}')


    #generate_and_save_grids(outdir, row_images, col_images, style_indices, include_originals, W, H)




#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
