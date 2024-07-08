import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

root_directory = 'videos'
for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(subdir, file)
                print(f"Processing video: {video_path}")
                
                # Load video
                video = os.path.basename(video_path)
                cap = cv2.VideoCapture(video_path)

                # Construct the CSV file path with the same directory and video name, but with .csv extension
                csv_filename = os.path.splitext(video_path)[0] + '.csv'
    

                # Open CSV file for writing
                csv_file = open(csv_filename, 'w', newline='')
                csv_writer = csv.writer(csv_file)


                # Define joint names
                joint_names = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
                            'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
                            'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
                            'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

                # Write header to CSV file
                header = ['Frame']
                for joint in joint_names:
                    header.extend([f'{joint}_X', f'{joint}_Y'])
                csv_writer.writerow(header)

                # Initialize drawing utility
                mp_drawing = mp.solutions.drawing_utils

                frame_count = 0  # To keep track of the frame number

                while cap.isOpened():
                    # Read a frame from the video
                    ret, frame = cap.read()
                    if not ret:
                        break
    
                    # Convert the frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the frame with MediaPipe Pose
                    results = pose.process(frame_rgb)
                    
                    # Extract and write the skeleton data to CSV
                    if results.pose_landmarks:
                        row_data = [frame_count]
                        for landmark, joint_name in zip(results.pose_landmarks.landmark, joint_names):
                            row_data.extend([landmark.x, landmark.y])
                        csv_writer.writerow(row_data)
                    
                    # Display the frame
                    cv2.imshow('Video', frame)
                    
                    # Increment frame count
                    frame_count += 1
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Release resources
                cap.release()
                cv2.destroyAllWindows()
                csv_file.close()
