import os
import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(frame1, frame2, win_size=7):
    # Compute the Structural Similarity Index (SSI) between two frames
    similarity_index, _ = ssim(frame1, frame2, full=True, win_size=win_size,channel_axis=2)
    return similarity_index

def remove_similar_frames(input_folder, output_folder, similarity_threshold=0.8, win_size=7):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create an empty list to store dictionaries for each video
    video_info_list = []

    # Iterate through all .avi files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".avi"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the input video file
            cap = cv2.VideoCapture(input_path)

            # Check if the video file is opened successfully
            if not cap.isOpened():
                print(f"Error: Couldn't open the video file {input_path}. Skipping.")
                continue

            # Read all frames into a list, skipping even frames
            frames = []
            discarded_frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % 2 != 0:
                    frames.append(frame)
                else:
                    discarded_frames.append(idx)
                idx += 1

            # Calculate SSIM with the middle frame
            middle_frame_index = len(frames) // 2
            middle_frame = frames[middle_frame_index]
            ssim_values = [calculate_ssim(middle_frame, frame, win_size) for frame in frames]

            # Rank frames based on SSIM
            ranked_frames = np.argsort(ssim_values)

            # Select 40% frames with the lowest similarity
            num_frames_to_select = int(0.4 * len(frames))
            selected_frames = set(ranked_frames[:num_frames_to_select])

            # Append the remaining 60% of the skipped frames
            discarded_frames += [i for i in range(len(frames)) if i not in selected_frames]

            # Create the VideoWriter object for the output video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frames[0].shape[1], frames[0].shape[0]), isColor=True)

            # Write selected frames to the output video
            for frame in selected_frames:
                out.write(frames[frame])

            # Release the video capture and writer objects
            cap.release()
            out.release()

            print(f"40% frames with lowest SSIM selected for {filename}. Output saved to {output_path}")

            # Append information to the list
            video_info_list.append({'Filename': filename, 'SkippedFrames': discarded_frames})

    # Create a DataFrame from the list
    video_info_df = pd.DataFrame(video_info_list)

    # Export the DataFrame to a CSV file
    csv_output_path = os.path.join(output_folder, 'combined_info.csv')
    video_info_df.to_csv(csv_output_path, index=False)
    print(f"Combined information exported to {csv_output_path}")

input_folder_path = "test"
output_folder_path = "colored"
remove_similar_frames(input_folder_path, output_folder_path, similarity_threshold=0.6, win_size=7)