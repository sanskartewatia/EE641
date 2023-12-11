import cv2
import numpy as np
from sklearn.cluster import KMeans
import random
import os
import pandas as pd
import time

def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def frame_to_feature(frame):
    """Convert a frame to a feature vector."""
    resized = cv2.resize(frame, (64, 64))  # Resize for faster processing
    feature = resized.flatten()
    return feature


def cluster_frames(frames, skip_rate):
    """Cluster frames and evenly extract frames from each cluster."""
    if not frames:
        print("No frames extracted from the video.")
        return []

    features = np.array([frame_to_feature(frame) for frame in frames])
    n_clusters = int(len(frames) * (1 - skip_rate))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    
    labels = kmeans.labels_
    
    # Calculate the number of frames to select from each cluster
    frames_per_cluster = 1

    selected_frames = []
    for i in range(n_clusters):
        cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
        selected_frames.extend(
            random.sample(
                cluster_indices, min(frames_per_cluster, len(cluster_indices))
            )
        )
    
    # Sort the selected frame indices and get the frames
    selected_frames.sort()
    representative_frames = [frames[index] for index in selected_frames]

    return representative_frames, selected_frames


def save_video(video, representative_frames, selected_frames, output_path):
    if not representative_frames:
        print("No frames selected, video will not be saved.")
        return
    height, width, _ = representative_frames[0].shape
    size = (width, height)
    video_path = os.path.join(output_path, 'videos')
    csv_path = os.path.join(output_path, 'indices')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
        
    codec = cv2.VideoWriter_fourcc(*"DIVX")  # Use DIVX for AVI
    out = cv2.VideoWriter(os.path.join(video_path, video), codec, 25, size)
    
    for frame in representative_frames:
        out.write(frame)
    out.release()
    
    name = video.split('.')[0]
    csv_name = f'{name}.csv'
    saved_frames_pd = pd.DataFrame(selected_frames)
    saved_frames_pd.to_csv(os.path.join(csv_path, csv_name), index=False)
    print(f"Video saved to: {output_path}. Save Frames: {selected_frames}")
    
start_time = time.time()
val_path = 'val'
output_path = 'KNN_0.8'
skip_rate = 0.8
for video in os.listdir(val_path):
    frames = extract_all_frames(os.path.join(val_path, video))
    representative_frames, selected_frames = cluster_frames(frames, skip_rate)
    save_video(video, representative_frames, selected_frames, output_path)
end_time = time.time()
print(f"Time: {end_time - start_time}")