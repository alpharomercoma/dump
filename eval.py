import os
import cv2
import numpy as np
import requests
import tensorflow as tf
import argparse

def download_video(url, save_path):
    """
    Downloads a video from the given URL and saves it to 'save_path'.
    Uses streaming download to handle large files.
    """
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download video: HTTP {response.status_code}")

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return save_path

def load_video(video_path, num_frames=16, target_size=(224, 224)):
    """
    Loads a video from disk, samples 'num_frames' uniformly,
    resizes frames to 'target_size', converts from BGR to RGB,
    and normalizes pixel values to [0,1].
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise Exception("Could not read any frames from the video.")

    # If video has fewer frames than desired, repeat the last frame.
    if total_frames < num_frames:
        frame_indices = np.concatenate([np.arange(total_frames),
                                        np.full((num_frames - total_frames,), total_frames - 1)])
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Use a blank frame if the read fails.
            frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)

def main():
    parser = argparse.ArgumentParser(description='Manually evaluate the video classification model.')
    parser.add_argument('video_url', type=str, help='URL of the video to evaluate')
    args = parser.parse_args()

    # Parameters for processing
    num_frames = 16
    target_size = (224, 224)
    temp_video_file = "temp_video.mp4"

    # Download the video from the provided URL.
    print("Downloading video...")
    download_video(args.video_url, temp_video_file)

    # Load and preprocess the video frames.
    print("Processing video...")
    video_frames = load_video(temp_video_file, num_frames=num_frames, target_size=target_size)
    # Expand dimensions to create a batch of size 1.
    video_frames = np.expand_dims(video_frames, axis=0)

    # Load the trained model.
    print("Loading model...")
    model = tf.keras.models.load_model('final_video_classifier.h5')

    # Make a prediction.
    print("Predicting...")
    pred = model.predict(video_frames)
    confidence = pred[0][0]
    label = "Sludge" if confidence >= 0.5 else "Non-Sludge"
    print(f"Video classification: {label}")
    print(f"Confidence (probability of sludge): {confidence:.4f}")

    # Clean up the temporary downloaded video file.
    if os.path.exists(temp_video_file):
        os.remove(temp_video_file)

if __name__ == '__main__':
    main()
