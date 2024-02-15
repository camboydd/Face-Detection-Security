import cv2
import os
import dlib
import numpy as np

# Function to load known faces
def load_known_faces():
    known_faces = {}
    
    # Iterate over directories in the known_faces_dir
    for person_folder in os.listdir(known_faces_dir):
        # Construct the full path to the person's folder
        person_folder_path = os.path.join(known_faces_dir, person_folder)
        
        # Check if the item in the directory is indeed a directory
        if os.path.isdir(person_folder_path):
            # Initialize list to store images of this person
            images = []
            # Iterate over files in the person's folder
            for filename in os.listdir(person_folder_path):
                # Check if the file is an image
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder_path, filename)
                    face_img = cv2.imread(img_path)
                    images.append(face_img)
            
            # Add the list of images to the dictionary with person's name as key
            known_faces[person_folder] = images
    
    return known_faces

# Define function to blur a region of interest
def blur_face(image, startX, startY, endX, endY):
    # Extract the ROI for the face
    face = image[startY:endY, startX:endX]
    
    # Apply Gaussian blur to the face ROI
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    
    # Replace the original face with the blurred face
    image[startY:endY, startX:endX] = blurred_face

# Initialize Dlib's facial landmark detector
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load face detection model
face_detector = dlib.get_frontal_face_detector()

# Load face recognition model
face_recognizer = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Directory containing folders of images of known faces
known_faces_dir = 'faces/'

# Load known faces
known_faces = load_known_faces()

# Start webcam
video_capture = cv2.VideoCapture(1)

# Process every 5th frame
frame_count = 0
skip_frames = 5

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame_count % skip_frames == 0:

        # Check if the frame is not empty
        if ret:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_detector(gray)

            # Iterate over detected faces
            for face in faces:
                # Get the coordinates of the bounding box of the face
                startX = face.left()
                startY = face.top()
                endX = face.right()
                endY = face.bottom()

                # Get the facial landmarks for face region
                landmarks = predictor(gray, face)

                # Align and crop the face using the facial landmarks
                aligned_face = dlib.get_face_chip(frame, landmarks, size=150)

                # Convert aligned face to RGB (required by Dlib)
                aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

                # Compute face descriptor (embedding)
                face_descriptor = face_recognizer.compute_face_descriptor(aligned_face_rgb)

                # Perform face recognition (compare with known faces)
                recognized = False

                # Threshold for face recognition (adjust as needed)
                threshold = 0.7

                # Iterate over items of the known_faces dictionary
                for name, images in known_faces.items():
                    for known_face in images:
                        # Convert known face to RGB (required by Dlib)
                        known_face_rgb = cv2.cvtColor(known_face, cv2.COLOR_BGR2RGB)

                        # Get facial landmarks for known face
                        known_landmarks = predictor(known_face_rgb, dlib.rectangle(0, 0, known_face.shape[1], known_face.shape[0]))

                        # Align and crop the known face using its facial landmarks
                        aligned_known_face = dlib.get_face_chip(known_face_rgb, known_landmarks, size=150)

                        # Convert aligned known face to RGB (required by Dlib)
                        aligned_known_face_rgb = cv2.cvtColor(aligned_known_face, cv2.COLOR_BGR2RGB)

                        # Compute known face descriptor (embedding)
                        known_face_descriptor = face_recognizer.compute_face_descriptor(aligned_known_face_rgb)

                        # Calculate Euclidean distance between face descriptors
                        distance = np.linalg.norm(np.array(face_descriptor) - np.array(known_face_descriptor))

                        # Check if the distance is below the threshold
                        if distance < threshold:
                            recognized = True
                            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            break
                    if recognized:
                        break

                # Draw rectangle around the face
                color = (255, 0, 0) if recognized else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Check if the face is not recognized
                if not recognized:
                    # Blur the face region
                    blur_face(frame, startX, startY, endX, endY)
                    
            # Display the resulting frame
            cv2.imshow('Video', frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()