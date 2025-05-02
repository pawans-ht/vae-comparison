---
title: Vae Comparison
emoji: 👀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
license: mit
short_description: Compare latest VAE's
---

## Command-Line Video Processor (`process_video_cli.py`)

A command-line tool to process videos through a VAE (Variational Autoencoder) and generate comparison outputs for each frame.

### Setup

1. Install UV
   ```
   https://docs.astral.sh/uv/getting-started/installation/
   ```
2. Install the required dependencies:
   ```bash
   uv sync
   ```

### Usage

The script provides a command-line interface with the following arguments:

- `--video_path` or `-v` (required): Path to the input video file
- `--output_dir` or `-o` (optional): Directory to save output grid images. If not specified, creates a directory named `output_frames_[video_name]`
- `--tolerance` or `-t` (optional): Difference tolerance threshold (default: 0.1)

#### Examples

1. Using long-form flags:
```bash
uv run process_video_cli.py --video_path path/to/video.mp4 --output_dir path/to/output --tolerance 0.1
```

2. Using short-form flags:
```bash
uv run process_video_cli.py -v path/to/video.mp4 -o path/to/output -t 0.1
```

3. Minimal usage (uses default output directory and tolerance):
```bash
uv run process_video_cli.py -v path/to/video.mp4
```

### Output

The script processes each frame of the video and creates a comparison grid image containing:
- Original frame
- Reconstructed frame (after VAE encoding/decoding)
- Difference visualization (highlighting areas that exceed the tolerance threshold)

Output images are saved as PNG files in the specified output directory, named sequentially as `frame_XXXX.png`.