def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print(f"Error: Couldn't open the video file {video_path}.")
        return None

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = frame_count / frame_rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size_bytes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * width * height * 3  # Assuming 3 channels (BGR)

    # Release the video capture object
    cap.release()

    # Return video information as a dictionary
    return {
        'frame_rate': frame_rate,
        'frame_count': frame_count,
        'duration_sec': duration_sec,
        'width': width,
        'height': height,
        'size_bytes': size_bytes
    }

def get_average_video_info(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.avi')]

    if not files:
        print(f"No AVI files found in the folder {folder_path}.")
        return None

    # Initialize cumulative values
    total_frame_rate = 0
    total_frame_count = 0
    total_duration_sec = 0
    total_width = 0
    total_height = 0
    total_size_bytes = 0

    # Iterate over all AVI files in the folder
    for filename in files:
        video_path = os.path.join(folder_path, filename)
        video_info = get_video_info(video_path)

        if video_info:
            total_frame_rate += video_info['frame_rate']
            total_frame_count += video_info['frame_count']
            total_duration_sec += video_info['duration_sec']
            total_width += video_info['width']
            total_height += video_info['height']
            total_size_bytes += video_info['size_bytes']

    # Calculate averages
    num_videos = len(files)
    average_frame_rate = total_frame_rate / num_videos
    average_frame_count = total_frame_count / num_videos
    average_duration_sec = total_duration_sec / num_videos
    average_width = total_width / num_videos
    average_height = total_height / num_videos
    average_size_bytes = total_size_bytes / num_videos

    # Return average video information as a dictionary
    return {
        'average_frame_rate': average_frame_rate,
        'average_frame_count': average_frame_count,
        'average_duration_sec': average_duration_sec,
        'average_width': average_width,
        'average_height': average_height,
        'average_size_bytes': average_size_bytes
    }


folder_path = "raw/val/"
average_video_info = get_average_video_info(folder_path)

if average_video_info:
    print("Average Video Information:")
    print(f"  Frame Rate: {average_video_info['average_frame_rate']} frames per second")
    print(f"  Number of Frames: {average_video_info['average_frame_count']}")
    print(f"  Average Duration: {average_video_info['average_duration_sec']:.2f} seconds")
    print(f"  Average Resolution: {int(average_video_info['average_width'])} x {int(average_video_info['average_height'])} pixels")
    print(f"  Average Size: {average_video_info['average_size_bytes'] / (1024 * 1024):.2f} MB")