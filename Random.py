import cv2
import random
import torch


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


def process_frames(frames, selection_percentage):
    frame_count = len(frames)
    selected_frame_count = int(frame_count * selection_percentage)

    # Generate a list of indices and shuffle it
    indices = list(range(frame_count))
    random.shuffle(indices)

    # Select the first N indices and sort them to preserve temporal order
    selected_indices = sorted(indices[:selected_frame_count])

    # Use the sorted indices to select frames
    selected_frames = [frames[i] for i in selected_indices]
    return selected_frames, selected_indices


def save_video(frames, output_path):
    if not frames:
        print("No frames selected, video will not be saved.")
        return
    height, width, layers = frames[0].shape
    size = (width, height)

    if output_path.endswith(".avi"):
        codec = cv2.VideoWriter_fourcc(*"DIVX")  # Use DIVX for AVI
    else:
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # Default to mp4v for other formats

    out = cv2.VideoWriter(output_path, codec, 15, size)

    for frame in frames:
        out.write(frame)
    out.release()


def randomSelection(video_path, output_path, selection_percentage=0.2):
    frames = load_video(video_path)
    selected_frames, selected_indices = process_frames(frames, selection_percentage)
    save_video(selected_frames, output_path)
    return selected_indices