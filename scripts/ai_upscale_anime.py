import argparse
import os
import subprocess
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
import math
import sys

# Import the centralized logger
from centralized_logger import logger, log_info, log_warning, log_error, log_debug

# Initialize colorama
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print("Please install colorama for colored CLI output: pip install colorama")
    sys.exit(1)

def define_model(model_name):
    if model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,  # Correct number of convolution layers
            upscale=4,    # Model's native upscale factor
            act_type='prelu'
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

def compute_scaling_factors(desired_scale):
    """Compute the sequence of scaling factors (1x, 2x, 3x, or 4x) to reach the desired scale."""
    scaling_options = [4, 3, 2, 1]
    current_scale = 1.0
    passes = []
    while current_scale < desired_scale:
        remaining_scale = desired_scale / current_scale
        # Choose the largest scaling factor that does not overshoot
        for scale in scaling_options:
            if current_scale * scale <= desired_scale or scale == 1:
                passes.append(scale)
                current_scale *= scale
                break
    return passes, current_scale

def inference_frames(frame_paths, output_dir, upsampler, scale_factor, device):
    """Upscale frames and save them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = len(frame_paths)
    pbar = tqdm(total=num_frames, desc='Upscaling Frames', unit='frame', ncols=80)

    batch_size = 8  # Adjust based on your GPU memory

    for i in range(0, num_frames, batch_size):
        batch_paths = frame_paths[i:i+batch_size]
        images = []

        for fp in batch_paths:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Failed to read image {fp}")
                continue
            images.append(img)

        if not images:
            continue  # Skip if images list is empty

        # Convert images to tensors and move to device
        imgs_tensor = [torch.from_numpy(img.transpose(2, 0, 1)).float().to(device) / 255.0 for img in images]
        imgs_tensor = torch.stack(imgs_tensor)

        # Adjust input data type based on model precision
        param_dtype = next(upsampler.model.parameters()).dtype
        if param_dtype == torch.float16:
            imgs_tensor = imgs_tensor.half()
        else:
            imgs_tensor = imgs_tensor.float()

        try:
            with torch.no_grad():
                outputs = upsampler.model(imgs_tensor)
            outputs = outputs.clamp_(0, 1).cpu().numpy()

            for idx, output in enumerate(outputs):
                output_img = (output * 255.0).astype(np.uint8)
                output_img = output_img.transpose(1, 2, 0)
                save_path = os.path.join(output_dir, os.path.basename(batch_paths[idx]))
                cv2.imwrite(save_path, output_img)
        except RuntimeError as error:
            logger.error(f'Error during upscaling frames {batch_paths}: {error}')
            continue

        pbar.update(len(batch_paths))

    pbar.close()
    logger.info(f"Upscaling process completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video file')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output directory')
    parser.add_argument('-n', '--model_name', type=str, default='realesr-animevideov3', help='Model name')
    parser.add_argument('-s', '--scale', type=float, default=2, help='Desired scaling factor (e.g., 1, 2, 3, 4, 6, 8)')
    parser.add_argument('--tile', type=int, default=0, help='Tile size (0 for no tiling)')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference')
    args = parser.parse_args()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info(Fore.GREEN + "Using CUDA (GPU) for upscaling")
    else:
        logger.info(Fore.YELLOW + "Using CPU for upscaling")

    # Load the model
    model_name = args.model_name
    model_path = os.path.join('weights', f'{model_name}.pth')

    if not os.path.isfile(model_path):
        # Download the model
        model_url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/{model_name}.pth'
        logger.info(f'Downloading model {model_name}')
        load_file_from_url(model_url, model_dir='weights')

    # Define the model
    model = define_model(model_name)

    # Load the model weights directly
    loadnet = torch.load(model_path, map_location=device)

    # Debugging: print available keys
    logger.info(f"Available keys in loaded model: {loadnet.keys()}")

    # Load state dict
    if 'params_ema' in loadnet:
        model.load_state_dict(loadnet['params_ema'], strict=True)
    elif 'params' in loadnet:
        model.load_state_dict(loadnet['params'], strict=True)
    else:
        model.load_state_dict(loadnet, strict=True)

    model.to(device)
    model.eval()

    # Prepare directories
    input_video = args.input
    output_dir = args.output
    input_frames_dir = os.path.join(output_dir, 'input_frames')
    final_output_dir = os.path.join(output_dir, 'upscaled_frames')

    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    # Extract frames from video
    extract_cmd = [
        'ffmpeg', '-i', input_video, '-vsync', '0', '-q:v', '1',
        os.path.join(input_frames_dir, 'frame%08d.png')
    ]
    logger.info('Extracting frames using ffmpeg...')
    subprocess.run(extract_cmd, check=True)

    frame_paths = sorted([os.path.join(input_frames_dir, f) for f in os.listdir(input_frames_dir) if f.endswith('.png')])

    # Compute the scaling passes based on the desired scale
    passes, final_scale = compute_scaling_factors(args.scale)
    remaining_scale = args.scale / final_scale

    logger.info(f"Scaling passes to be applied: {passes}")
    logger.info(f"Final scale after passes: {final_scale}")
    logger.info(f"Remaining scale factor to reach desired scale: {remaining_scale}")

    # Upscale frames according to the passes
    for idx, scale in enumerate(passes):
        logger.info(f'Upscaling pass {idx + 1}/{len(passes)} with scale factor {scale}x')
        pass_output_dir = os.path.join(output_dir, f'upscaled_pass_{idx + 1}')
        os.makedirs(pass_output_dir, exist_ok=True)

        # Initialize upsampler with the current scale
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,  # Set to None since we're passing the model directly
            dni_weight=None,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            device=device
        )

        # Process frames
        inference_frames(frame_paths, pass_output_dir, upsampler, scale_factor=scale, device=device)

        # Update frame paths for the next pass
        frame_paths = sorted([os.path.join(pass_output_dir, f) for f in os.listdir(pass_output_dir) if f.endswith('.png')])

    # Handle remaining scale if necessary
    if remaining_scale != 1.0:
        logger.info(f'Remaining scale factor: {remaining_scale}')
        os.makedirs(final_output_dir, exist_ok=True)

        for fp in tqdm(frame_paths, desc='Adjusting Remaining Scale', unit='frame', ncols=80):
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            new_h, new_w = int(h * remaining_scale), int(w * remaining_scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            save_path = os.path.join(final_output_dir, os.path.basename(fp))
            cv2.imwrite(save_path, img_resized)
    else:
        # Move the upscaled frames to the final output directory
        logger.info(f"Final pass: saving upscaled frames to {final_output_dir}")
        for fp in frame_paths:
            save_path = os.path.join(final_output_dir, os.path.basename(fp))
            os.rename(fp, save_path)  # Move frames to final output directory

    logger.info(f'Upscaling completed. Upscaled frames are saved in {final_output_dir}')

if __name__ == '__main__':
    main()
