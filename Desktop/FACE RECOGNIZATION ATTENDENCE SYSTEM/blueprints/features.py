from flask import Blueprint, render_template, session, send_file, jsonify, redirect, url_for, request, Response
import sqlite3
import os
from flask import send_from_directory
import cv2
import pandas as pd
from blueprints.data_train import train_model  # Import the train_model function
import numpy as np
from threading import Thread
from .detect import start_detection, release_video_stream

features_bp = Blueprint('features', __name__)

# Global variable to control the video stream
video_stream = None

@features_bp.route('/')
def features():
    return render_template('features.html')

@features_bp.route('/studentdata',)
def student_data():
    return render_template('studentdata.html')

@features_bp.route('/trainData', methods=['POST'])
def traindata():
    try:
        train_model()  # Call the training function directly
        return jsonify({"message": "Data trained successfully!"}), 200
    except Exception as e:
        print(f"Error during training: {e}")  # Print the error output
        return jsonify({"error": "Failed to train data."}), 500

@features_bp.route('/startAttendance', methods=['GET'])
def start_attendance():
    # Fetch all teachers and subjects from the database
    conn = sqlite3.connect('sqlite.db')
    conn.row_factory = sqlite3.Row
    teachers = conn.execute("SELECT DISTINCT Name FROM teachers").fetchall()
    subjects = conn.execute("SELECT DISTINCT Subject FROM teachers").fetchall()
    conn.close()
    return render_template('start_attendance.html', teachers=teachers, subjects=subjects)

@features_bp.route('/api/startNewAttendance', methods=['POST'])
def api_start_new_attendance():
    global video_stream
    # Check if the video stream is already initialized
    if video_stream is None or not video_stream.isOpened():
        url = "http://10.255.203.77:4747/video"  # URL of the depth camera
        video_stream = cv2.VideoCapture(url)  # Start the video stream with default camera index

    # Check if the video stream was opened successfully
    if not video_stream.isOpened():
        print("Error: Could not open video stream.")
        return jsonify({"error": "Could not open video stream."}), 500

    # Get the selected data from the form
    teacher = request.form.get('teacher')
    subject = request.form.get('subject')
    period = request.form.get('period')

    # Store the session data for video feed
    session['teacher'] = teacher
    session['subject'] = subject
    session['period'] = period

    # Start the detection process in a separate thread
    detection_thread = Thread(target=start_detection, args=(teacher, subject, period, video_stream))
    detection_thread.start()

    # Redirect to the attendance page
    return redirect(url_for('features.attendance'))

@features_bp.route('/video_feed')
def video_feed():
    global video_stream
    if video_stream is None or not video_stream.isOpened():
        return "Video stream not initialized.", 500

    # Call the start_detection function to get processed frames
    detection_generator = start_detection(session.get('teacher'), session.get('subject'), session.get('period'), video_stream)

    def generate():
        try:
            for frame in detection_generator:
                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Failed to encode frame.")
                    break
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in video feed generation: {e}")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@features_bp.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global video_stream
    if video_stream is not None:
        release_video_stream(video_stream)  # Release the camera
        video_stream = None  # Reset the global variable
    return redirect(url_for('features.features'))  # Redirect to the features page

@features_bp.route('/attendance')
def attendance():
    return render_template('attendance.html')

@features_bp.route('/teacherData')
def teacher_data():
    conn = sqlite3.connect('sqlite.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Subject FROM teachers")
    teacher_subject_info = cursor.fetchall()
    conn.close()
    
    # Debugging: Print the fetched data
    print("Fetched teacher_subject_info:", teacher_subject_info)
    
    return render_template('teacher_subject_info.html', teacher_subject_info=teacher_subject_info)
    
@features_bp.route('/admin_dashboard')
def admin_dashboard():
    # Check if the admin is logged in
    if 'admin_id' in session:
        return render_template('admin_dashboard.html', admin_id=session['admin_id'])
    else:
        return "Access denied. Please log in as an admin.", 403

@features_bp.route('/show_attendance', methods=['GET', 'POST'])
def show_attendance():
    conn = sqlite3.connect('sqlite.db')
    conn.row_factory = sqlite3.Row  # Fetch rows as dictionaries
    cursor = conn.cursor()

    # Initialize filters
    filters = {}
    if request.method == 'POST':
        # Get filter values from the form
        date = request.form.get('date')
        teacher = request.form.get('teacher')
        subject = request.form.get('subject')
        period = request.form.get('period')

        # Build the SQL query based on filters
        query = "SELECT * FROM ATTENDANCE WHERE 1=1"
        if date:
            query += f" AND date = '{date}'"
            filters['date'] = date
        if teacher:
            query += f" AND teacher = '{teacher}'"
            filters['teacher'] = teacher
        if subject:
            query += f" AND subject = '{subject}'"
            filters['subject'] = subject
        if period:
            query += f" AND period = '{period}'"
            filters['period'] = period

        # Execute the query
        cursor.execute(query)
        attendance_records = cursor.fetchall()
    else:
        # If no filters are applied, fetch all records
        cursor.execute("SELECT * FROM ATTENDANCE")
        attendance_records = cursor.fetchall()

    # Fetch unique teachers, subjects, and periods for the filter dropdowns
    teachers = conn.execute("SELECT DISTINCT teacher FROM ATTENDANCE").fetchall()
    subjects = conn.execute("SELECT DISTINCT subject FROM ATTENDANCE").fetchall()
    periods = conn.execute("SELECT DISTINCT period FROM ATTENDANCE").fetchall()

    conn.close()

    # Render the template with attendance records and filters
    return render_template('show_attendance.html', 
                           attendance_records=attendance_records, 
                           teachers=teachers, 
                           subjects=subjects, 
                           periods=periods, 
                           filters=filters)

@features_bp.route('/download_attendance', methods=['POST'])
def download_attendance():
    conn = sqlite3.connect('sqlite.db')
    cursor = conn.cursor()

    # Get filter values from the form
    date = request.form.get('date')
    teacher = request.form.get('teacher')
    subject = request.form.get('subject')

    # Build the SQL query based on filters
    query = """
    SELECT a.student_id, s.roll, a.date, a.time, a.status, a.teacher, a.subject, a.period
    FROM ATTENDANCE a
    JOIN students s ON a.student_id = s.Id
    WHERE a.date = ? AND a.teacher = ? AND a.subject = ?
    """
    cursor.execute(query, (date, teacher, subject))
    attendance_records = cursor.fetchall()

    # Convert records to a CSV file
    import csv
    import os

    # Define the directory to save the CSV file
    save_directory = r"C:\Users\Junaid\Desktop\FACE RECOGNIZATION ATTENDENCE SYSTEM\Attendance Records"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create a descriptive file name
    file_name = f"Attendance_{teacher}_{subject}_{date}.csv"
    file_path = os.path.join(save_directory, file_name)

    # Write the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID', 'Roll', 'Date', 'Time', 'Status', 'Teacher', 'Subject', 'Period'])  # Header
        for record in attendance_records:
            writer.writerow(record)

    conn.close()

    # Return the CSV file as a downloadable response
    return send_from_directory(save_directory, file_name, as_attachment=True)

@features_bp.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        date = data.get('date')
        time = data.get('time')
        teacher = data.get('teacher')
        subject = data.get('subject')
        period = data.get('period')

        print(f"Received data: student_id={student_id}, date={date}, time={time}, teacher={teacher}, subject={subject}, period={period}")

        conn = sqlite3.connect('sqlite.db')
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM attendance 
            WHERE student_id = ? AND date = ? AND time = ? AND teacher = ? AND subject = ? AND period = ?
        """, (student_id, date, time, teacher, subject, period))
        conn.commit()
        print(f"Deleted rows: {cursor.rowcount}")
        conn.close()
        return jsonify({"status": "success", "message": "Attendance record deleted successfully."})
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({"status": "error", "message": f"Database error: {e}"})