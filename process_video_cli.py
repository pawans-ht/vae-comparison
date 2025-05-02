import argparse
import os
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
import torchvision.transforms.v2 as transforms
from PIL import Image

class VAETester:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.input_transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.base_transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.output_transform = transforms.Normalize(mean=[-1], std=[2])
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)

    def process_frame(self, frame: np.ndarray, tolerance: float):
        """Process a single video frame through the VAE"""
        # Convert BGR to RGB and normalize to [0, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) / 255.0

        img_transformed = self.input_transform(frame_tensor).to(self.device).unsqueeze(0)
        original_base = self.base_transform(frame_tensor).cpu()

        with torch.no_grad():
            encoded = self.vae.encode(img_transformed).latent_dist.sample()
            decoded = self.vae.decode(encoded).sample

        decoded_transformed = self.output_transform(decoded.squeeze(0)).cpu()
        reconstructed = decoded_transformed.clip(0, 1)
        diff = (original_base - reconstructed).abs()
        bw_diff = (diff > tolerance).any(dim=0).float()
        diff_image = transforms.ToPILImage()(bw_diff)
        recon_image = transforms.ToPILImage()(reconstructed)
        original_image = transforms.ToPILImage()(original_base)
        return original_image, recon_image, diff_image


def create_comparison_grid(original, reconstructed, diff):
    """Create a horizontal grid of original, reconstructed, and difference images"""
    # Ensure all images are PIL Images
    if not isinstance(original, Image.Image):
        original = transforms.ToPILImage()(original)
    if not isinstance(reconstructed, Image.Image):
        reconstructed = transforms.ToPILImage()(reconstructed)
    if not isinstance(diff, Image.Image):
        diff = transforms.ToPILImage()(diff)

    # Resize images to have the same height if necessary, maintaining aspect ratio
    max_h = max(original.height, reconstructed.height, diff.height)
    if original.height != max_h:
        original = original.resize((int(original.width * max_h / original.height), max_h))
    if reconstructed.height != max_h:
        reconstructed = reconstructed.resize((int(reconstructed.width * max_h / reconstructed.height), max_h))
    if diff.height != max_h:
        diff = diff.resize((int(diff.width * max_h / diff.height), max_h))

    # Convert diff to RGB if it's grayscale
    if diff.mode == 'L' or diff.mode == '1':
        diff = diff.convert('RGB')

    total_width = original.width + reconstructed.width + diff.width
    grid = Image.new('RGB', (total_width, max_h))
    grid.paste(original, (0, 0))
    grid.paste(reconstructed, (original.width, 0))
    grid.paste(diff, (original.width + reconstructed.width, 0))
    return grid


def process_video(video_path: str, output_dir: str, tolerance: float = 0.1):
    """Process a video file and save comparison grid images for each frame"""
    # Verify input video exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {video_path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate padding for frame numbers
        padding = len(str(total_frames))

        # Initialize VAE model
        vae_tester = VAETester()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            print(f"\rProcessing frame {frame_idx + 1}/{total_frames}", end="", flush=True)
            
            original, reconstructed, diff = vae_tester.process_frame(frame, tolerance)
            grid = create_comparison_grid(original, reconstructed, diff)

            # Save the grid image
            output_path = os.path.join(output_dir, f"frame_{frame_idx:0{padding}d}.png")
            grid.save(output_path)

            frame_idx += 1

        print(f"\nProcessed {frame_idx} frames. Output saved to: {output_dir}")

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Process video frames through VAE and save comparison grids")
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", default=None, help="Path to save output grid images (optional)")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Difference tolerance (default: 0.1)")

    args = parser.parse_args()

    # Create default output directory name if none provided
    if args.output_dir is None:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        args.output_dir = f"output_frames_{video_name}"

    try:
        process_video(args.video_path, args.output_dir, args.tolerance)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()