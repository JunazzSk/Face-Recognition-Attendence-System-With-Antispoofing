import cv2
import numpy as np
import sqlite3
from datetime import datetime
import dlib  # For face detection and embeddings
import joblib
import os
from ultralytics import YOLO  # Import YOLO for anti-spoofing

# Dynamically determine the absolute path to the model files
model_path = os.path.join(os.path.dirname(__file__), "../models/dlib_face_recognition_resnet_model_v1.dat")
landmark_model_path = os.path.join(os.path.dirname(__file__), "../models/shape_predictor_68_face_landmarks.dat")
knn_model_path = os.path.join(os.path.dirname(__file__), "../models/face_recognition_knn.pkl")
label_encoder_path = os.path.join(os.path.dirname(__file__), "../models/label_encoder.pkl")
yolo_model_path = os.path.join(os.path.dirname(__file__), "../models/best.pt")  # Path to YOLO model

# Load dlib models
print(f"Loading models from: {model_path} and {landmark_model_path}")
try:
    face_rec_model = dlib.face_recognition_model_v1(model_path)
    shape_predictor = dlib.shape_predictor(landmark_model_path)
    print("Dlib models loaded successfully!")
except RuntimeError as e:
    print(f"Error loading dlib models: {e}")
    exit(1)  # Exit the program if the models fail to load

# Load YOLO anti-spoofing model
try:
    yolo_model = YOLO(yolo_model_path)
    print("YOLO anti-spoofing model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Function to get face embedding using dlib
def get_embedding(face_image, face_rect):
    # Detect facial landmarks
    landmarks = shape_predictor(face_image, face_rect)

    # Align the face using dlib.get_face_chip
    aligned_face = dlib.get_face_chip(face_image, landmarks, size=150)

    # Get the face embedding
    embedding = face_rec_model.compute_face_descriptor(aligned_face)
    # Normalize the embedding for consistent scaling
    embedding = np.array(embedding) / np.linalg.norm(embedding)
    return embedding

# Function to detect faces using dlib
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces
    face_boxes = [{'box': (face.left(), face.top(), face.width(), face.height()), 'rect': face} for face in faces]
    return face_boxes

# Function to check if the entire frame is real or spoofed using YOLO
def is_real_frame_yolo(frame, model):
    """
    Check if the entire frame is real or spoofed using the YOLO model.
    :param frame: The input frame (numpy array)
    :param model: The YOLO model
    :return: (bool, str) True if the frame is real, False otherwise, and the label ("real" or "spoof")
    """
    # Ensure the frame is a NumPy array
    if not isinstance(frame, np.ndarray):
        raise ValueError("Input frame must be a NumPy array.")

    # Convert the frame to RGB (YOLO expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform prediction
    results = model(rgb_frame, verbose=False)

    # Process the results
    for result in results:
        if hasattr(result, 'probs') and result.probs is not None:  # Ensure 'probs' exists and is not None
            probs = result.probs  # Probs object for classification outputs
            label = model.names[probs.top1]  # Get the predicted label (e.g., "real" or "spoof")
            if label == "real":
                return True, "real"  # Real frame detected
            elif label == "spoof":
                return False, "spoof"  # Spoofed frame detected

    return False, "spoof"  # Default to spoofed if no result

# Function to fetch student profile from the database
def get_profile(student_id):
    print(f"Querying profile for student ID: {student_id}")
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE Id=?", (student_id,))
    profile = cursor.fetchone()
    conn.close()
    if profile is None:
        print(f"No profile found for student ID: {student_id}")
    return profile

# Function to mark attendance in the database
def mark_attendance(student_id, teacher, subject, period):
    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    current_time = datetime.now().strftime("%H:%M:%S")  # Format: HH:MM:SS

    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()

    # Check if attendance already exists for the given student, teacher, subject, period, and date
    cursor.execute(
        """
        SELECT * FROM ATTENDANCE
        WHERE student_id = ? AND teacher = ? AND subject = ? AND period = ? AND date = ?
        """,
        (student_id, teacher, subject, period, current_date)
    )
    existing_record = cursor.fetchone()

    if existing_record:
        print(f"Attendance already marked for student ID: {student_id} on {current_date} for Teacher: {teacher}, Subject: {subject}, Period: {period}")
    else:
        # Insert attendance record into the ATTENDANCE table
        cursor.execute(
            """
            INSERT INTO ATTENDANCE (student_id, date, time, status, teacher, subject, period)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, current_date, current_time, "Present", teacher, subject, period)
        )
        conn.commit()
        print(f"Attendance marked for student ID: {student_id} on {current_date} at {current_time}")

    conn.close()

# Main loop for face detection and recognition
def start_detection(teacher, subject, period, video_stream):
    # Load the KNN classifier and label encoder
    model_dir = os.path.join(os.path.dirname(__file__), "../models")
    classifier_path = os.path.join(model_dir, 'face_recognition_knn.pkl')
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

    if os.path.exists(classifier_path) and os.path.exists(label_encoder_path):
        classifier = joblib.load(classifier_path)
        label_encoder = joblib.load(label_encoder_path)
    else:
        print("Classifier and label encoder files not found. Please train the model first.")
        return

    print(f"Label Mapping: {list(label_encoder.classes_)}")
    print(f"Starting attendance detection for Teacher: {teacher}, Subject: {subject}, Period: {period}")

    while True:
        ret, img_depth = video_stream.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Reduce frame resolution to improve performance
        img_depth = cv2.resize(img_depth, (640, 480))

        # Perform YOLO-based anti-spoofing check on the entire frame
        is_real, label = is_real_frame_yolo(img_depth, yolo_model)

        # Display the label on the video feed
        color = (0, 255, 0) if is_real else (0, 0, 255)  # Green for "real", Red for "spoof"
        cv2.putText(img_depth, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not is_real:
            print("Spoofed frame detected, skipping...")
            yield img_depth  # Yield the frame with the "spoof" label
            continue  # Skip the frame if it's spoofed

        # Perform face detection only if the frame is classified as "real"
        faces = detect_faces(img_depth)
        print(f"Detected {len(faces)} faces")

        for face in faces:
            x, y, w, h = face['box']
            cropped_face = img_depth[y:y + h, x:x + w]  # Crop the face region

            try:
                embedding = get_embedding(img_depth, face['rect'])
                predicted_label = classifier.predict([embedding])[0]
                confidence_scores = classifier.predict_proba([embedding])[0]
                confidence = max(confidence_scores)

                predicted_id = int(label_encoder.inverse_transform([predicted_label])[0])
                print(f"Predicted Label: {predicted_label}, Student ID: {predicted_id}, Confidence: {confidence}")

                if confidence >= 0.4:
                    profile = get_profile(predicted_id)
                    if profile:
                        print(f"Profile found: {profile}")
                        mark_attendance(predicted_id, teacher, subject, period)
                        student_name = profile[1]
                        student_roll_no = profile[2]
                        cv2.putText(img_depth, f"Name: {student_name}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(img_depth, f"Roll No: {student_roll_no}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"No profile found for student ID: {predicted_id}")
                else:
                    print(f"Low confidence for predicted ID: {predicted_id}, skipping...")

            except Exception as e:
                print(f"Error occurred during attendance detection: {e}")

        # Yield the processed frame for the video feed
        yield img_depth

        # Add a small delay to limit frame rate
        cv2.waitKey(1)

    video_stream.release()
    cv2.destroyAllWindows()
    print("Video stream released.")

# Function to release the video stream when done
def release_video_stream(video_stream):
    if video_stream is not None:
        video_stream.release()
        print("Video stream released.")