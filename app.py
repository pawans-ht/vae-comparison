import spaces
import gradio as gr
import torch
from diffusers import AutoencoderKL
from diffusers.utils.remote_utils import remote_decode
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image
from typing import Dict
import os
from huggingface_hub import login


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
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", img_size: int = 512):
        self.device = device
        self.input_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.base_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size)),
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
        """Load configurations for local and remote VAE models"""
        local_vaes = {
            "stable-diffusion-v1-4": AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device),
            "sd-vae-ft-mse": AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device),
            "sdxl-vae": AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(self.device),
            "stable-diffusion-3-medium": AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae").to(self.device),
            "FLUX.1-schnell": AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="vae").to(self.device),
            "FLUX.1-dev": AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae").to(self.device),
        }
        # Define the desired order of models
        order = [
            "stable-diffusion-v1-4",
            "sd-vae-ft-mse",
            "sd-vae-ft-mse (remote)",
            "sdxl-vae",
            "sdxl-vae (remote)",
            "stable-diffusion-3-medium",
            "FLUX.1-schnell",
            "FLUX.1-schnell (remote)",
            "FLUX.1-dev",
        ]

        # Construct the vae_models dictionary in the specified order
        vae_models = {}
        for name in order:
            if "(remote)" not in name:
                # Local model
                vae_models[name] = {"type": "local", "vae": local_vaes[name]}
            else:
                # Remote model
                base_name = name.replace(" (remote)", "")
                vae_models[name] = {
                    "type": "remote",
                    "local_vae_key": base_name,
                    "endpoint": self._get_endpoint(base_name),
                }

        return vae_models

    def process_image(self, img: torch.Tensor, model_config: Dict, tolerance: float):
        """Process image through a single VAE (local or remote)"""
        img_transformed = self.input_transform(img).to(self.device).unsqueeze(0)
        original_base = self.base_transform(img).cpu()

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

    def process_all_models(self, img: torch.Tensor, tolerance: float):
        """Process image through all configured VAEs"""
        results = {}
        for name, model_config in self.vae_models.items():
            diff_img, recon_img, score = self.process_image(img, model_config, tolerance)
            results[name] = (diff_img, recon_img, score)
        return results

@spaces.GPU(duration=15)
def test_all_vaes(image_path: str, tolerance: float, img_size: int):
    """Gradio interface function to test all VAEs"""
    tester = VAETester(img_size=img_size)
    try:
        img_tensor = read_image(image_path)
        results = tester.process_all_models(img_tensor, tolerance)

        diff_images = []
        recon_images = []
        scores = []

        for name in tester.vae_models.keys():
            diff_img, recon_img, score = results[name]
            diff_images.append((diff_img, name))
            recon_images.append((recon_img, name))
            scores.append(f"{name:<25}: {score:,.0f}")

        return diff_images, recon_images, "\n".join(scores)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return [None], [None], error_msg

examples = [f"examples/{img_filename}" for img_filename in sorted(os.listdir("examples/"))]

with gr.Blocks(title="VAE Performance Tester", css=".monospace-text {font-family: 'Courier New', Courier, monospace;}") as demo:
    gr.Markdown("# VAE Comparison Tool")
    gr.Markdown("""
        Upload an image or select an example to compare how different VAEs reconstruct it. Now includes remote VAEs via Hugging Face's remote decoding feature!
        1. The image is padded to a square and resized to the selected size (512 or 1024 pixels).
        2. Each VAE (local or remote) encodes the image into a latent space and decodes it back.
        3. Outputs include:
           - **Difference Maps**: Where reconstruction differs from the original (white = difference > tolerance).
           - **Reconstructed Images**: Outputs from each VAE.
           - **Sum of Differences**: Total pixels exceeding tolerance (lower is better).
        Adjust tolerance to change sensitivity.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Image", height=512)
            tolerance_slider = gr.Slider(
                minimum=0.01,
                maximum=0.5,
                value=0.1,
                step=0.01,
                label="Difference Tolerance",
                info="Low (0.01): Sensitive to small changes. High (0.5): Only large changes flagged."
            )
            img_size = gr.Dropdown(label="Image Size", choices=[512, 1024], value=512)
            submit_btn = gr.Button("Test All VAEs")

        with gr.Column(scale=3):
            with gr.Row():
                diff_gallery = gr.Gallery(label="Difference Maps", columns=4, height=512)
                recon_gallery = gr.Gallery(label="Reconstructed Images", columns=4, height=512)
            scores_output = gr.Textbox(label="Sum of differences (lower is better)", lines=9, elem_classes="monospace-text")

    if examples:
        with gr.Row():
            gr.Examples(examples=examples, inputs=image_input, label="Example Images")

    submit_btn.click(
        fn=test_all_vaes,
        inputs=[image_input, tolerance_slider, img_size],
        outputs=[diff_gallery, recon_gallery, scores_output]
    )

if __name__ == "__main__":
    demo.launch()
