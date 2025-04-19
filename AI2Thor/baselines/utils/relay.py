import cv2
import numpy as np
import os


def combine_frames_and_save_video(frames1, frames2, output_path, fps=30):
    """
    Combines two sets of frames of the same size into a single video, playing them side by side.

    Args:
        frames1 (list): List of frames (numpy arrays) for the first video.
        frames2 (list): List of frames (numpy arrays) for the second video.
        output_path (str): Path to save the combined video.
        fps (int): Frames per second for the output video.
    """
    if len(frames1) != len(frames2):
        raise ValueError("Both frame sets must have the same number of frames.")

    # Get frame dimensions
    height, width, channels = frames1[0].shape

    # Define the codec and create VideoWriter object
    combined_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))

    for frame1, frame2 in zip(frames1, frames2):
        # Combine frames side by side
        combined_frame = np.hstack((frame1, frame2))
        out.write(combined_frame)

    out.release()
    print(f"Video saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Directory paths for the two sets of frames
    dir1 = "AI2Thor/baselines/results/llamar/render_images/1_put_bread_lettuce_tomato_fridge/FloorPlan1/2/Alice/pov"
    dir2 = "AI2Thor/baselines/results/llamar/render_images/1_put_bread_lettuce_tomato_fridge/FloorPlan1/2/Bob/pov"

    # Get sorted lists of image file paths
    frames1_files = sorted(
        [
            os.path.join(dir1, f)
            for f in os.listdir(dir1)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    frames2_files = sorted(
        [
            os.path.join(dir2, f)
            for f in os.listdir(dir2)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Ensure both directories have the same number of frames
    if len(frames1_files) != len(frames2_files):
        raise ValueError("Both directories must contain the same number of frames.")

    # Load frames
    frames1 = [cv2.imread(f) for f in frames1_files]
    frames2 = [cv2.imread(f) for f in frames2_files]

    # Ensure frames are valid
    if not all(frame is not None for frame in frames1 + frames2):
        raise ValueError("Some frames could not be loaded.")

    combine_frames_and_save_video(frames1, frames2, "output_video.mp4")
