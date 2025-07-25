import os
import cv2
import numpy as np
import sqlite3
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import dlib
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dlib models
model_path = r"C:\Users\Junaid\Desktop\FACE RECOGNIZATION ATTENDENCE SYSTEM\models\dlib_face_recognition_resnet_model_v1.dat"
landmark_model_path = r"C:\Users\Junaid\Desktop\FACE RECOGNIZATION ATTENDENCE SYSTEM\models\shape_predictor_68_face_landmarks.dat"

logging.info(f"Loading models from: {model_path} and {landmark_model_path}")
try:
    face_rec_model = dlib.face_recognition_model_v1(model_path)
    shape_predictor = dlib.shape_predictor(landmark_model_path)
    logging.info("Models loaded successfully!")
except RuntimeError as e:
    logging.error(f"Error loading models: {e}")
    exit(1)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Function to get face embedding using dlib
def get_embedding(face_image, face_rect):
    landmarks = shape_predictor(face_image, face_rect)
    aligned_face = dlib.get_face_chip(face_image, landmarks, size=150)
    embedding = face_rec_model.compute_face_descriptor(aligned_face)
    return np.array(embedding)

# Function to detect faces using dlib
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_boxes = [{'box': (face.left(), face.top(), face.width(), face.height())} for face in faces]
    return face_boxes

# Function to train the model using KNN
def train_model():
    image_folder = "dataset"
    embeddings = []
    labels = []

    # Connect to the database
    try:
        conn = sqlite3.connect('sqlite.db')
        cursor = conn.cursor()
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        return

    # Fetch student data
    cursor.execute("SELECT Id, Name FROM students")
    students = cursor.fetchall()

    for student in students:
        student_id = student[0]
        student_name = student[1]
        formatted_id = f"{student_id:02d}"

        image_files = [f for f in os.listdir(image_folder) if f.startswith(f"{formatted_id}_{student_name}") and f.endswith('.jpg')]

        if not image_files:
            logging.warning(f"No images found for student {student_name} (ID: {student_id}).")
            continue

        for image_name in image_files:
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                logging.warning(f"Failed to load image: {image_name}")
                continue

            faces = detect_faces(image)
            if len(faces) == 0:
                logging.warning(f"No faces detected in image: {image_name}")
                continue

            x, y, w, h = faces[0]['box']
            face_rect = dlib.rectangle(x, y, x + w, y + h)

            embedding = get_embedding(image, face_rect)
            if embedding.size == 0:
                logging.warning(f"Invalid embedding for image: {image_name}")
                continue

            embeddings.append(embedding.tolist())
            labels.append(student_id)

    if len(embeddings) == 0 or len(labels) == 0:
        logging.error("No data available for training. Please check the dataset.")
        conn.close()
        return

    model_dir = r"C:\Users\Junaid\Desktop\FACE RECOGNIZATION ATTENDENCE SYSTEM\models"
    os.makedirs(model_dir, exist_ok=True)

    embeddings_path = os.path.join(model_dir, 'embeddings.npy')
    labels_path = os.path.join(model_dir, 'labels.npy')
    np.save(embeddings_path, np.array(embeddings))
    np.save(labels_path, np.array(labels))

    unique_labels = set(labels)
    logging.info(f"Unique labels found: {unique_labels}")

    if len(unique_labels) > 1:
        # Encode labels as integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Perform grid search with cross-validation
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=3)
        grid_search.fit(np.array(embeddings), encoded_labels)

        # Get the best estimator
        best_knn = grid_search.best_estimator_
        logging.info(f"Best Parameters: {grid_search.best_params_}")

        # Save the trained KNN classifier and label encoder
        classifier_path = os.path.join(model_dir, 'face_recognition_knn.pkl')
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(best_knn, classifier_path)
        joblib.dump(label_encoder, label_encoder_path)
        logging.info("KNN model trained and saved successfully.")

    else:
        logging.warning("Not enough classes to train the model. Saved embeddings and labels for future training.")
 
    conn.close()
    
