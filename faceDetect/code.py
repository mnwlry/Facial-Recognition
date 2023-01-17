import face_recognition
import dlib
import cv2
import numpy as np
import os
import glob
import time
from datetime import datetime

# Load known faces
known_faces = []
path = 'known_faces/'
for filename in glob.glob(path + '*.jpg'):
    image = face_recognition.load_image_file(filename)
    # check if there is at least one face in the image
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings)>0:
        encoding = face_encodings[0]
        known_faces.append(encoding)
    else:
        print(f"No face detected in {filename}")

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = os.path.splitext(os.path.basename(path + glob.glob(path + '*.jpg')[first_match_index]))[0]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    font = cv2.FONT_HERSHEY_DUPLEX
    if name == "Unknown":
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Save the full-size captured image of the unknown face
        filename = "Captured/unknown_"+datetime.now().strftime("%Y%m%d-%H%M%S")+".jpg"
        cv2.imwrite(filename, frame)
        
                
    elif name != "Unknown":
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)
        # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    


