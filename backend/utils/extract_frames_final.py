import cv2
import os

def extract_frames(video_path, output_dir, frames_per_video=3):
    """Extract evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"No frames in video: {video_path}")
        return 0

    count = 0
    for i in range(frames_per_video):
        frame_idx = int((i + 1) * total_frames / (frames_per_video + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = f'{os.path.splitext(os.path.basename(video_path))[0]}_frame_{i+1}.jpg'
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            count += 1
            print(f"Extracted frame {i+1} from {os.path.basename(video_path)}")
    cap.release()
    return count

def main():
    # Create directories
    os.makedirs('data/processed/train/real', exist_ok=True)
    os.makedirs('data/processed/train/fake', exist_ok=True)

    # Extract from real videos
    real_count = 0
    real_dir = 'Test/Real'
    if os.path.exists(real_dir):
        for video in os.listdir(real_dir):
            if video.endswith('.mp4'):
                video_path = os.path.join(real_dir, video)
                real_count += extract_frames(video_path, 'data/processed/train/real')
                print(f"Processed real video: {video}")

    # Extract from fake videos
    fake_count = 0
    fake_dir = 'Test/deepfake'
    if os.path.exists(fake_dir):
        for video in os.listdir(fake_dir):
            if video.endswith('.mp4'):
                video_path = os.path.join(fake_dir, video)
                fake_count += extract_frames(video_path, 'data/processed/train/fake')
                print(f"Processed fake video: {video}")

    print(f'\nExtraction complete!')
    print(f'Extracted {real_count} real frames and {fake_count} fake frames')

if __name__ == "__main__":
    main()