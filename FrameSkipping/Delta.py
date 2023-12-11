import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time

def write_video(saved_frames, input_path, output_path):
    video_path = os.path.join(output_path, 'videos')
    csv_path = os.path.join(output_path, 'indices')
    cap = cv2.VideoCapture(input_path)
    video_name = input_path.split('\\')[1]
    name = video_name.split('.')[0]
    csv_name = f'{name}.csv'

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    out = cv2.VideoWriter(os.path.join(video_path, video_name), cv2.VideoWriter_fourcc(*"DIVX"), frame_rate, (frame_width, frame_height))

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in saved_frames:
            out.write(frame)

        current_frame += 1
    
    cap.release()
    out.release()
    saved_frames_pd = pd.DataFrame(saved_frames)
    saved_frames_pd.to_csv(os.path.join(csv_path, csv_name), index=False)
    print(f"Video saved to: {output_path}. Save Frames: {saved_frames}. Skip rate: {len(saved_frames) / frame_cnt}")

def remove_similar_frames(input_folder, output_folder, skip_rate, win_size=7):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all .avi files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".avi"):
            input_path = os.path.join(input_folder, filename)

            # Open the input video file
            cap = cv2.VideoCapture(input_path)

            # Check if the video file is opened successfully
            if not cap.isOpened():
                print(f"Error: Couldn't open the video file {input_path}. Skipping.")
                continue
            frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            saved_cnt = int((1 - skip_rate) * frame_cnt)
            # Get the first frame
            _, prev_frame = cap.read()
            # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            similarity_vec = []
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                delta = np.abs(frame - prev_frame)
                similarity_index = np.sum(delta) / delta.size

                # Check if the frames are dissimilar enough
                similarity_vec.append(similarity_index)
                prev_frame = frame
                
            indices = np.argsort(similarity_vec)
            saved_frames = indices[:saved_cnt]
            saved_frames.sort()
            saved_frames += 1
            write_video(saved_frames, input_path, output_folder)

start_time = time.time()
input_folder_path = "val"
output_folder_path = "delta_80"
skip_rate = 0.8
remove_similar_frames(input_folder_path, output_folder_path, skip_rate, win_size=7)
end_time = time.time()
print(f"Time: {end_time - start_time}")