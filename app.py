import gradio as gr
import torch
from diffusers import AutoencoderKL
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image
from typing import Tuple, Dict, List
import os
from huggingface_hub import login

# Get token from environment variable
hf_token = os.getenv("access_token")
login(token=hf_token)

class VAETester:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.input_transform = transforms.Compose([
            transforms.Pad(padding=[128, 0], padding_mode="edge"),
            transforms.Resize((512, 512), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.base_transform = transforms.Compose([
            transforms.Pad(padding=[128, 0], padding_mode="edge"),
            transforms.Resize((512, 512), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.output_transform = transforms.Normalize(mean=[-1], std=[2])

        # Load all VAE models at initialization
        self.vae_models = self._load_all_vaes()

    def _load_all_vaes(self) -> Dict[str, AutoencoderKL]:
        """Load all available VAE models"""
        vae_configs = {
            "Stable Diffusion 3 Medium": ("stabilityai/stable-diffusion-3-medium-diffusers", "vae"),
            "Stable Diffusion v1-4": ("CompVis/stable-diffusion-v1-4", "vae"),
            "SD VAE FT-MSE": ("stabilityai/sd-vae-ft-mse", ""),
            "FLUX.1-dev": ("black-forest-labs/FLUX.1-dev", "vae")
        }

        vae_dict = {}
        for name, (path, subfolder) in vae_configs.items():
            vae_dict[name] = AutoencoderKL.from_pretrained(path, subfolder=subfolder).to(self.device)
        return vae_dict

    def process_image(self,
                      img: torch.Tensor,
                      vae: AutoencoderKL,
                      tolerance: float):
        """Process image through a single VAE"""
        img_transformed = self.input_transform(img).to(self.device).unsqueeze(0)
        original_base = self.base_transform(img).cpu()

        with torch.no_grad():
            encoded = vae.encode(img_transformed).latent_dist.sample()
            encoded_scaled = encoded * vae.config.scaling_factor
            decoded = vae.decode(encoded_scaled / vae.config.scaling_factor).sample

        decoded_transformed = self.output_transform(decoded.squeeze(0)).cpu()
        reconstructed = decoded_transformed.clip(0, 1)

        diff = (original_base - reconstructed).abs()
        bw_diff = (diff > tolerance).any(dim=0).float()

        diff_image = transforms.ToPILImage()(bw_diff)
        recon_image = transforms.ToPILImage()(reconstructed)
        diff_score = bw_diff.sum().item()

        return diff_image, recon_image, diff_score

    def process_all_models(self,
                           img: torch.Tensor,
                           tolerance: float):
        """Process image through all loaded VAEs"""
        results = {}
        for name, vae in self.vae_models.items():
            diff_img, recon_img, score = self.process_image(img, vae, tolerance)
            results[name] = (diff_img, recon_img, score)
        return results


# Initialize tester
tester = VAETester()


def test_all_vaes(image_path: str, tolerance: float):
    """Gradio interface function to test all VAEs"""
    try:
        img_tensor = read_image(image_path)
        results = tester.process_all_models(img_tensor, tolerance)

        diff_images = []
        recon_images = []
        scores = []

        for name in tester.vae_models.keys():
            diff_img, recon_img, score = results[name]
            diff_images.append(diff_img)
            recon_images.append(recon_img)
            scores.append(f"{name}: {score:.2f}")

        return diff_images, recon_images, scores

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return [None], [None], [error_msg]


# Gradio interface
with gr.Blocks(title="VAE Performance Tester") as demo:
    gr.Markdown("# VAE Performance Testing Tool")
    gr.Markdown("Upload an image to compare all VAE models simultaneously")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Image", height=512)
            tolerance_slider = gr.Slider(
                minimum=0.01,
                maximum=0.5,
                value=0.1,
                step=0.01,
                label="Difference Tolerance"
            )
            submit_btn = gr.Button("Test All VAEs")

        with gr.Column(scale=3):
            with gr.Row():
                diff_gallery = gr.Gallery(label="Difference Maps", columns=4, height=512)
                recon_gallery = gr.Gallery(label="Reconstructed Images", columns=4, height=512)
            scores_output = gr.Textbox(label="Difference Scores", lines=4)

    submit_btn.click(
        fn=test_all_vaes,
        inputs=[image_input, tolerance_slider],
        outputs=[diff_gallery, recon_gallery, scores_output]
    )

if __name__ == "__main__":
    demo.launch()
