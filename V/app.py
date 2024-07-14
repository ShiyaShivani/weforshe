import cv2
import mediapipe as mp
import numpy as np
import time
from utils import overlay_clothing, load_clothing_images
from flask import Flask, render_template, url_for, request
import threading

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load clothing images
clothing_images = load_clothing_images('clothing/')
current_clothing_index = 0
last_change_time = time.time()  # Track the last time the clothing was changed


app = Flask(__name__)

@app.route('/')

@app.route('/home')
def home():
    return render_template("Index.html")

@app.route('/main',methods=['POST'])

def run():
    # output = request.form.to_dict()
    # print(output)
    # name = output["name"]
    # main()
    thread = threading.Thread(target=main)
    thread.start()
    # return render_template("Index.html")
    return "<h1>TRY ON RUNNING!!</h1>"

    
def main():
    global current_clothing_index, last_change_time
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find pose landmarks
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = {point: [int(results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, point)].x * frame.shape[1]),
                                 int(results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark, point)].y * frame.shape[0])]
                        for point in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_WRIST', 'RIGHT_WRIST']}
            
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check for hand-raising to change clothes
            current_time = time.time()
            if current_time - last_change_time > 2:  # Ensure at least 2 seconds have passed
                if keypoints['LEFT_WRIST'][1] < keypoints['LEFT_SHOULDER'][1]:
                    current_clothing_index = (current_clothing_index - 1) % len(clothing_images)
                    last_change_time = current_time
                elif keypoints['RIGHT_WRIST'][1] < keypoints['RIGHT_SHOULDER'][1]:
                    current_clothing_index = (current_clothing_index + 1) % len(clothing_images)
                    last_change_time = current_time
            
            # Overlay clothing on the frame
            clothing_image = clothing_images[current_clothing_index]
            overlay_clothing(frame, clothing_image, keypoints)
        
        # Draw buttons on the frame
        cv2.putText(frame, 'Previous', (50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Next', (frame.shape[1] - 150, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Virtual Try-On', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # main()
    app.run(debug=True, use_reloader=False)
