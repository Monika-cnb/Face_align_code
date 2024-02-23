# Code for video face-alignment
import os
import pandas as pd
import dlib
import cv2

# all files list
all_file_list = ["DG25032013.avi", "MP04022013.avi"]

input_file_path = "./../../"
output_file_path = "./../../"


for each_video in all_file_list:
    print(each_video)
    # file_path = input_file_path + each_file
    # Load the shape predictor model
    shape_predictor_path = "./../../shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(shape_predictor_path)
    # Load the face detector
    detector = dlib.get_frontal_face_detector()
    # Load the video
    video_path = input_file_path + each_video
    print(video_path)
    # read each video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # reading the output file
    output_path = output_file_path + each_video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
    left, right, top, bottom = 0, 0, 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Conversion to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect face
        faces = detector(gray)
        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)
            # Get bounding box for facial coordinates
            left, top, right, bottom = (max(face.left(), 0), max(face.top(), 0), min(face.right(), frame.shape[1]),
                                        min(face.bottom(), frame.shape[0]))
            print("All co-ordinate value: left, right, top, bottom", left, right, top, bottom)
            break
        break

    old_left, old_top, old_right, old_bottom = max(left-100, 0), max(top-100, 0), min(right+100, frame.shape[1]), min(bottom+100, frame.shape[0])
    print("All updated coordinate values: old_left, old_right, old_top, old_bottom", old_left, old_right, old_top, old_bottom)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            # Crop the aligned face region
            aligned_face_region = frame[old_top:old_bottom, old_left:old_right]
            aligned_face_region_resized = cv2.resize(aligned_face_region, (frame_width, frame_height))
            # print("aligned_face_region shape:", aligned_face_region.shape)
            # print("aligned_face_region dtype:", aligned_face_region.dtype)
            # print("actual face shape:", frame.shape)
            out.write(aligned_face_region_resized)
            ###out.write(frame)

            # Display the cropped face region
            # cv2.imshow("Cropped Face", aligned_face_region)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Write the cropped face region to the ImageWriter
    cap.release()
    cv2.destroyAllWindows()