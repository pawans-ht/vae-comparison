import spaces
import gradio as gr
import torch
from diffusers import AutoencoderKL
from diffusers.utils.remote_utils import remote_decode
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image
from typing import Dict, Tuple
import os
import cv2
import numpy as np
from huggingface_hub import login
from PIL import Image

# Get token from environment variable
hf_token = os.getenv("access_token")
login(token=hf_token)

class PadToSquare:
    """Custom transform to pad an image to square dimensions"""
    def __call__(self, img):
        _, h, w = img.shape  # Get the original dimensions
        max_side = max(h, w)
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2
        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        return transforms.functional.pad(img, padding, padding_mode="edge")

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
        self.vae_models = self._load_all_vaes()

    def _get_endpoint(self, base_name: str) -> str:
        """Helper method to get the endpoint for a given base model name"""
        endpoints = {
            "sd-vae-ft-mse": "https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud",
            "sdxl-vae": "https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud",
            "FLUX.1-schnell": "https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud",
        }
        return endpoints[base_name]

    def _load_all_vaes(self) -> Dict[str, Dict]:
        """Load only the sd-vae-ft-mse model configuration"""
        local_vaes = {
            "sd-vae-ft-mse": AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        }
        vae_models = {
            "sd-vae-ft-mse": {"type": "local", "vae": local_vaes["sd-vae-ft-mse"]}
        }
        return vae_models

    def process_frame(self, frame: np.ndarray, model_config: Dict, tolerance: float):
        """Process a single video frame through a VAE"""
        # Convert BGR to RGB and normalize to [0, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) / 255.0
        
        img_transformed = self.input_transform(frame_tensor).to(self.device).unsqueeze(0)
        original_base = self.base_transform(frame_tensor).cpu()

        if model_config["type"] == "local":
            vae = model_config["vae"]
            with torch.no_grad():
                encoded = vae.encode(img_transformed).latent_dist.sample()
                decoded = vae.decode(encoded).sample
        elif model_config["type"] == "remote":
            local_vae = self.vae_models[model_config["local_vae_key"]]["vae"]
            with torch.no_grad():
                encoded = local_vae.encode(img_transformed).latent_dist.sample()
            decoded = remote_decode(
                endpoint=model_config["endpoint"],
                tensor=encoded,
                do_scaling=False,
                output_type="pt",
                return_type="pt",
                partial_postprocess=False,
            )
        
        decoded_transformed = self.output_transform(decoded.squeeze(0)).cpu()
        reconstructed = decoded_transformed.clip(0, 1)
        diff = (original_base - reconstructed).abs()
        bw_diff = (diff > tolerance).any(dim=0).float()
        diff_image = transforms.ToPILImage()(bw_diff)
        recon_image = transforms.ToPILImage()(reconstructed)
        diff_score = bw_diff.sum().item()
        return diff_image, recon_image, diff_score

    def process_all_models(self, frame: np.ndarray, tolerance: float):
        """Process a video frame through all configured VAEs"""
        results = {}
        for name, model_config in self.vae_models.items():
            diff_img, recon_img, score = self.process_frame(frame, model_config, tolerance)
            results[name] = (diff_img, recon_img, score)
        return results

def get_video_info(video_path: str) -> Tuple[int, int]:
    """Get total frame count and FPS of the video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return total_frames, fps

@spaces.GPU(duration=15)
def create_comparison_grid(original, reconstructed, diff):
    """Create a horizontal grid of original, reconstructed, and difference images"""
    width = max(original.width, reconstructed.width, diff.width)
    height = max(original.height, reconstructed.height, diff.height)
    grid = Image.new('RGB', (width * 3, height))
    grid.paste(original, (0, 0))
    grid.paste(reconstructed, (width, 0))
    grid.paste(diff, (width * 2, 0))
    return grid

def test_all_vaes(video_path: str, start_frame: int, end_frame: int, tolerance: float):
    """Process multiple frames using sd-vae-ft-mse and create comparison grids"""
    if not video_path:
        return [], "Error: No video file provided"

    try:
        total_frames, fps = get_video_info(video_path)
        
        # Validate frame range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        tester = VAETester()
        
        comparison_grids = []
        scores = []
        
        # Process each frame in the range
        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB and to PIL Image for original
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(frame_rgb)
            
            # Process frame through VAE
            results = tester.process_all_models(frame, tolerance)
            diff_img, recon_img, score = results["sd-vae-ft-mse"]
            
            # Create comparison grid
            grid = create_comparison_grid(original_pil, recon_img, diff_img)
            comparison_grids.append((grid, f"Frame {frame_num}"))
            scores.append(f"Frame {frame_num}: {score:,.0f}")
        
        cap.release()
        frame_info = f"\nProcessed frames {start_frame}-{end_frame} of {total_frames}"
        return comparison_grids, "\n".join(scores) + frame_info

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return [], error_msg

with gr.Blocks(title="Video VAE Performance Tester", css=".monospace-text {font-family: 'Courier New', Courier, monospace;}") as demo:
    gr.Markdown("# Video Frame VAE Analysis Tool")
    gr.Markdown("""
        Upload a video file to analyze frames using the sd-vae-ft-mse model.
        1. Select a video file and specify the frame range to process.
        2. All frames in the range will be processed.
        3. For each frame, you'll see:
           - **Comparison Grid**: Original | Reconstructed | Difference Map
           - **Difference Score**: Total pixels exceeding tolerance (lower is better)
        Adjust tolerance to change sensitivity of difference detection.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            start_frame = gr.Number(label="Start Frame", value=0, precision=0)
            end_frame = gr.Number(label="End Frame", value=10, precision=0)
            tolerance_slider = gr.Slider(
                minimum=0.01,
                maximum=0.5,
                value=0.1,
                step=0.01,
                label="Difference Tolerance",
                info="Low (0.01): Sensitive to small changes. High (0.5): Only large changes flagged."
            )
            submit_btn = gr.Button("Process Frames")

        with gr.Column(scale=3):
            comparison_gallery = gr.Gallery(
                label="Frame Comparisons (Original | Reconstructed | Difference)",
                columns=2,
                height=512
            )
            scores_output = gr.Textbox(
                label="Difference scores by frame",
                lines=10,
                elem_classes="monospace-text"
            )

    submit_btn.click(
        fn=test_all_vaes,
        inputs=[video_input, start_frame, end_frame, tolerance_slider],
        outputs=[comparison_gallery, scores_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, ssr_mode=False)