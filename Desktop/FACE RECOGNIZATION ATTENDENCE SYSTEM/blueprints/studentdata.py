import os
import cv2
import numpy as np
import sqlite3
from flask import Blueprint, render_template, request, redirect, url_for, Response, jsonify
from PIL import Image
import time
import dlib  # Using dlib for face detection and alignment

# Define Blueprint
studentdata_bp = Blueprint('studentdata', __name__, template_folder='../templates')

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# Dictionary to track the completion status of dataset capture for each student
capture_status = {}

# Database interaction functions
def insert_or_update(Id, Name, Roll, Class):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE Id=?", (Id,))
    isRecordExist = cursor.fetchone() is not None

    if isRecordExist:
        cursor.execute(
            "UPDATE students SET Name=?, Roll=?, Class=? WHERE Id=?",
            (Name, Roll, Class, Id)
        )
        print(f"Updated record for ID: {Id}")
    else:
        cursor.execute(
            "INSERT INTO students (Id, Name, Roll, Class) VALUES (?, ?, ?, ?)",
            (Id, Name, Roll, Class)
        )
        print(f"Inserted new record for ID: {Id}")
    conn.commit()
    conn.close()

# Route to render the student data entry form
@studentdata_bp.route('/')
def student_data_entry_form():
    return render_template('studentdata.html')

# Route to handle student data entry
@studentdata_bp.route('/student_data_entry', methods=['POST'])
def student_data_entry():
    try:
        student_id = request.form['id']
        student_id = student_id.zfill(2)  # Convert student ID to string with leading zeros
        Name = request.form['name']
        Roll = request.form['roll']
        Class = request.form['class']

        insert_or_update(student_id, Name, Roll, Class)
        return redirect(url_for('studentdata.capture_dataset_page', student_id=student_id))
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Check server logs.", 500

# Route to render the dataset capture page
@studentdata_bp.route('/capture_dataset/<int:student_id>')
def capture_dataset_page(student_id):
    # Reset the capture status for the student ID
    capture_status[student_id] = False
    # Render the video feed page
    return render_template('capture_dataset.html', student_id=student_id)

# Route to handle video feed for dataset capture
@studentdata_bp.route('/video_feed/<int:student_id>')
def video_feed(student_id):
    def generate():
        url = "http://192.168.144.187:4747/video"  # URL of the depth camera
        cam_depth = cv2.VideoCapture(url)  # Open the depth camera
        if not cam_depth.isOpened():
            print("Error: Could not open the camera.")
            return

        sampleNum = 0

        # Get the student's name from the database
        conn = sqlite3.connect("sqlite.db")
        cursor = conn.cursor()
        cursor.execute("SELECT Name FROM students WHERE Id=?", (student_id,))
        student_name = cursor.fetchone()[0]
        conn.close()

        while sampleNum < 20:
            ret_depth, img_depth = cam_depth.read()
            if not ret_depth:
                print("Failed to access depth camera.")
                break

            # Detect faces using dlib
            gray = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            faces = detector(gray)  # Detect faces

            for face in faces:
                # Align the face using dlib's shape predictor and get_face_chip
                landmarks = shape_predictor(img_depth, face)
                aligned_face = dlib.get_face_chip(img_depth, landmarks, size=150)

                # Check for blurriness
                if is_blurry(aligned_face):
                    print("Blurry image detected, skipping...")
                    continue

                # Save the aligned face
                sampleNum += 1
                pil_image = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
                pil_image.save(f"dataset/{student_id:02d}_{student_name}_{sampleNum}.jpg")

                # Draw rectangle around the face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(img_depth, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_depth, f"Capturing Image {sampleNum}/20", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add a delay between captures
                time.sleep(0.5)  # 0.5 seconds delay

            ret, buffer = cv2.imencode('.jpg', img_depth)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cam_depth.release()
        cv2.destroyAllWindows()

        # Mark the capture as complete
        capture_status[student_id] = True

        # Send a completion signal
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\nCOMPLETE\r\n')
        print("Sent completion signal")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to check for blurriness
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Route to check the capture status
@studentdata_bp.route('/capture_status/<int:student_id>', methods=['GET'])
def capture_status_endpoint(student_id):
    # Check if the capture is complete for the given student ID
    if capture_status.get(student_id, False):
        return jsonify({"status": "COMPLETE"})
    return jsonify({"status": "IN_PROGRESS"})

# Main function to run the application
if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(studentdata_bp)
    app.run(debug=True)