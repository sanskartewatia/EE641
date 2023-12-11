import cv2
import random
import os
import pandas as pd
import time

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def process_frames(frames, skip_rate):
    frame_count = len(frames)
    selected_frame_count = int(frame_count * (1 - skip_rate))

    # Generate a list of indices and shuffle it
    indices = list(range(frame_count))
    random.shuffle(indices)

    # Select the first N indices and sort them to preserve temporal order
    selected_indices = sorted(indices[:selected_frame_count])

    # Use the sorted indices to select frames
    selected_frames = [frames[i] for i in selected_indices]
    return selected_frames, selected_indices


def save_video(frames, video, selected_indices, output_path):
    if not frames:
        print("No frames selected, video will not be saved.")
        return
    video_path = os.path.join(output_path, 'videos')
    csv_path = os.path.join(output_path, 'indices')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        
    height, width, _ = frames[0].shape
    size = (width, height)

    codec = cv2.VideoWriter_fourcc(*"DIVX")  # Use DIVX for AVI    

    out = cv2.VideoWriter(os.path.join(video_path, video), codec, 25, size)

    for frame in frames:
        out.write(frame)
    out.release()
    
    name = video.split('.')[0]
    csv_name = f'{name}.csv'
    saved_frames_pd = pd.DataFrame(selected_indices)
    saved_frames_pd.to_csv(os.path.join(csv_path, csv_name), index=False)
    print(f"Video saved to: {video_path}. Save Frames: {selected_indices}")

start_time = time.time()
val_path = 'val'
output_path = 'random_80'
skip_rate = 0.8
for video in os.listdir(val_path):
    frames = load_video(os.path.join(val_path, video))
    selected_frames, selected_indices = process_frames(frames, skip_rate)
    save_video(selected_frames, video, selected_indices, output_path)

end_time = time.time()
print(f"Time: {end_time - start_time}")