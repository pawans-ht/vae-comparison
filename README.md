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

**Important Note:** This tool requires `ffmpeg` to be installed separately on your system for video creation functionality.

### Usage

The script provides a command-line interface with the following arguments:

- `--video_path` or `-v` (required): Path to the input video file
- `--output_dir` or `-o` (optional): Directory to save output grid images. If not specified, creates a directory named `output_frames_[video_name]`
- `--tolerance` or `-t` (optional): Difference tolerance threshold (default: 0.1)
- `--create_video` or `-c` (optional): Flag to create output video

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

4. Creating a comparison video:
```bash
uv run process_video_cli.py -v path/to/video.mp4 -o path/to/output -c
```

### Output

The script processes each frame of the video and creates a comparison grid image containing:
- Original frame
- Reconstructed frame (after VAE encoding/decoding)
- Difference visualization (highlighting areas that exceed the tolerance threshold)

Output images are saved as PNG files in an `analysis_results` subdirectory within the specified output directory, named sequentially as `frame_XXXX.png`. When using the `-c` flag, these frames are also compiled into a video file using ffmpeg.

## Batch Video Processor (`batch_process_videos.py`)

A utility script to process multiple videos in a directory using the VAE comparison tool.

### Usage

The batch processor provides a command-line interface with the following arguments:

- `--videos_dir` or `-d` (required): Directory containing video files to process
- `--output_dir` or `-o` (optional): Base directory to save output for all videos
- `--tolerance` or `-t` (optional): Difference tolerance threshold (default: 0.1)
- `--create_video` or `-c` (optional): Flag to create comparison videos for all processed videos
- `--parallel` or `-p` (optional): Number of videos to process in parallel (default: 1)

#### Examples

1. Process all videos in a directory sequentially:
```bash
uv run batch_process_videos.py -d path/to/videos
```

2. Process videos with custom settings:
```bash
uv run batch_process_videos.py -d path/to/videos -o path/to/batch_output -t 0.15 -c
```

3. Process multiple videos in parallel (e.g., 4 at a time):
```bash
uv run batch_process_videos.py -d path/to/videos -o path/to/batch_output -c -p 4
```

### Output

The batch processor creates separate output directories for each video within the specified output directory. Each video's output follows the same structure described in the Command-Line Video Processor section.

The script provides progress updates during execution and a summary of successfully/failed processing attempts at the end.