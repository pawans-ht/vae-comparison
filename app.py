import gradio as gr
import torch
from diffusers import AutoencoderKL
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
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.input_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((512, 512), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.base_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((512, 512), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.output_transform = transforms.Normalize(mean=[-1], std=[2])

        # Load all VAE models at initialization
        self.vae_models = self._load_all_vaes()

    def _load_all_vaes(self) -> Dict[str, AutoencoderKL]:
        """Load all available VAE models"""
        vae_configs = {
            "stable-diffusion-v1-4": ("CompVis/stable-diffusion-v1-4", "vae"),
            "sd-vae-ft-mse": ("stabilityai/sd-vae-ft-mse", ""),
            "sdxl-vae": ("stabilityai/sdxl-vae", ""),
            "stable-diffusion-3-medium": ("stabilityai/stable-diffusion-3-medium-diffusers", "vae"),
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

@spaces.GPU(duration=10)
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
            diff_images.append((diff_img, name))
            recon_images.append((recon_img, name))
            scores.append(f"{name:<25}: {score:.1f}")

        return diff_images, recon_images, "\n".join(scores)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return [None], [None], error_msg

examples = [f"examples/{img_filename}" for img_filename in sorted(os.listdir("examples/"))]

# Gradio interface
with gr.Blocks(title="VAE Performance Tester", css=".monospace-text {font-family: 'Courier New', Courier, monospace;}") as demo:
    gr.Markdown("# VAE Comparison Tool")
    gr.Markdown("""
        Upload an image or select an example to compare how different VAEs reconstruct it. Here's what happens:
        1. The image is padded to a square and resized to 512x512 pixels.
        2. Each VAE encodes the image into a latent space and decodes it back.
        3. The tool then generates:
           - **Difference Maps**: Black-and-white images showing where the reconstruction differs from the original (white areas indicate differences above the tolerance threshold).
           - **Reconstructed Images**: The outputs from each VAE.
           - **Sum of Differences**: A numerical score for each VAE, measuring the total difference in pixels exceeding the tolerance.
        Use the tolerance slider to adjust the sensitivity.
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
                info="Low tolerance (e.g., 0.01): Highly sensitive, flags small deviations. High tolerance (e.g., 0.5): Less sensitive, flags only large deviations, showing fewer differences.",
            )
            submit_btn = gr.Button("Test All VAEs")

        with gr.Column(scale=3):
            with gr.Row():
                diff_gallery = gr.Gallery(label="Difference Maps", columns=4, height=512)
                recon_gallery = gr.Gallery(label="Reconstructed Images", columns=4, height=512)
            scores_output = gr.Textbox(label="Sum of difference (lower is better reconstruction)", lines=5, elem_classes="monospace-text")

        if examples:
            with gr.Column():
                example_gallery = gr.Examples(
                    examples=examples,
                    inputs=image_input,
                    label="Example Images"
                )

    submit_btn.click(
        fn=test_all_vaes,
        inputs=[image_input, tolerance_slider],
        outputs=[diff_gallery, recon_gallery, scores_output]
    )

if __name__ == "__main__":
    demo.launch()

