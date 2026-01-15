#!/usr/bin/env python3
"""
Extract frames from videos in Test folder and organize into training data
"""
import cv2
import os
from pathlib import Path

def extract_frames_from_videos(video_dir, output_dir, frames_per_video=5):
    """Extract frames from videos in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    frame_count = 0

    print(f'Processing {len(video_files)} videos from {video_dir}')

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f'Could not open {video_file}')
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract frames at regular intervals
        for i in range(frames_per_video):
            frame_idx = int((i + 1) * total_frames / (frames_per_video + 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_filename = f'{os.path.splitext(video_file)[0]}_frame_{i+1:02d}.jpg'
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1

        cap.release()

    print(f'Extracted {frame_count} frames to {output_dir}')
    return frame_count

def main():
    print("🎬 Extracting frames from videos...")

    # Extract frames from real videos
    real_frames = extract_frames_from_videos('Test/Real', 'data/processed/train/real')

    # Extract frames from fake videos
    fake_frames = extract_frames_from_videos('Test/deepfake', 'data/processed/train/fake')

    print(f"\n✅ Total frames extracted: {real_frames + fake_frames}")
    print("✅ Data ready for training!")

if __name__ == "__main__":
    main()