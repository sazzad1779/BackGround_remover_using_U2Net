import cv2
import numpy as np

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (you can choose a different one)
frame_size = (1280, 720)  # Width x Height
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, frame_size)

if not out.isOpened():
    print("Error: VideoWriter could not be opened.")
    exit()

# Generate some example frames
for i in range(100):
    # Create a sample frame with the desired shape (720, 1280, 3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame.fill(255)  # Fill with white color
    print(frame.shape)
    # Write the frame to the video
    out.write(frame)

# Release the video writer
out.release()
