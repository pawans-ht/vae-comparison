#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import concurrent.futures
from pathlib import Path


# Common video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv']


def is_video_file(file_path):
    """Check if a file has a video extension"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def process_single_video(video_path, args):
    """Process a single video using process_video_cli.py"""
    video_name = os.path.basename(video_path)
    print(f"\n{'='*80}")
    print(f"Processing video: {video_name}")
    print(f"{'='*80}")

    # Build the command for process_video_cli.py
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "process_video_cli.py"),
        "-v", video_path
    ]

    # Add output directory if specified
    if args.output_dir:
        video_output_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0])
        cmd.extend(["-o", video_output_dir])
    
    # Add tolerance if specified
    if args.tolerance is not None:
        cmd.extend(["-t", str(args.tolerance)])
    
    # Add create_video flag if specified
    if args.create_video:
        cmd.append("-c")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        return True, video_path
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False, video_path


def main():
    parser = argparse.ArgumentParser(description="Process multiple videos through VAE")
    parser.add_argument("-d", "--videos_dir", required=True, 
                        help="Directory containing video files to process")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Base directory to save output for all videos (optional)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.1,
                        help="Difference tolerance (default: 0.1)")
    parser.add_argument("-c", "--create_video", action="store_true",
                        help="Create comparison videos from processed frames")
    parser.add_argument("-p", "--parallel", type=int, default=1,
                        help="Number of videos to process in parallel (default: 1)")
    
    args = parser.parse_args()
    
    # Validate videos directory
    videos_dir = os.path.abspath(args.videos_dir)
    if not os.path.isdir(videos_dir):
        print(f"Error: Directory not found: {videos_dir}")
        return 1
    
    # Create output directory if specified and doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Find video files in the directory
    video_files = []
    for file in os.listdir(videos_dir):
        file_path = os.path.join(videos_dir, file)
        if os.path.isfile(file_path) and is_video_file(file_path):
            video_files.append(file_path)
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return 0
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process videos
    successful = 0
    failed = 0
    
    if args.parallel > 1:
        # Process videos in parallel
        print(f"Processing {len(video_files)} videos using {args.parallel} parallel workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_single_video, video, args): video for video in video_files}
            for future in concurrent.futures.as_completed(futures):
                success, video_path = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                print(f"Progress: {successful + failed}/{len(video_files)} completed")
    else:
        # Process videos sequentially
        print(f"Processing {len(video_files)} videos sequentially...")
        for i, video_path in enumerate(video_files, 1):
            print(f"Processing video {i}/{len(video_files)}")
            success, _ = process_single_video(video_path, args)
            if success:
                successful += 1
            else:
                failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Processing completed:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*80}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())