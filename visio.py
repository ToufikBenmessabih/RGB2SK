import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video
video_path = 'split1/P01_R01_v1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may need to try different codecs depending on your system
output_video_path = 'split1/skeleton_v1.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Extract and visualize the skeleton
    if results.pose_landmarks:
        # Draw the skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
