# Face-Recognition-Attendence-System-With-Antispoofing

This project is a real-time Face Recognition Attendance System designed for educational institutions and workplaces. It uses deep learning and computer vision to automate attendance marking, ensuring both convenience and security.

## Features

- **Real-Time Face Recognition:**  
  Detects and recognizes faces from a live camera feed using dlib and a KNN classifier.

- **Anti-Spoofing with YOLO:**  
  Integrates a YOLO-based anti-spoofing model to prevent fraudulent attendance using photos, videos, or masks.

- **Automated Attendance:**  
  Recognized faces are automatically marked as present in an SQLite database, along with the date, time, subject, teacher, and period.

- **Web Interface:**  
  Flask-based web app for managing attendance and student/teacher data.

- **Visualization:**  
  Tools to visualize face embedding distances and recognition results for system evaluation and tuning.

## How It Works

1. **Live Video Feed:** Captures frames from a webcam.
2. **Anti-Spoofing:** YOLO model checks if the face is real or spoofed.
3. **Face Detection & Recognition:** dlib detects faces and extracts embeddings, which are classified using KNN.
4. **Attendance Marking:** Recognized faces are marked present in the database.
5. **Admin Dashboard:** Manage and view attendance records.

## Technologies Used

- Python, OpenCV, dlib, scikit-learn, Flask
- YOLO (Ultralytics) for anti-spoofing
- SQLite for database

## Getting Started

1. Clone the repository.
2. Install dependencies (`pip install -r requirements.txt`).
3. Prepare your dataset (face images for each student).
4. Train the model using the provided scripts.
5. Run the Flask app and start marking attendance!

---

**This system is ideal for classrooms, offices, and events where secure, contactless, and automated attendance is required.**

Let me know if you want a shorter or more detailed version!
