from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import sqlite3
import os

admin_dashboard_bp = Blueprint('admin_dashboard', __name__, template_folder='../templates')

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('sqlite.db')
    conn.row_factory = sqlite3.Row
    return conn

# Admin Dashboard Route
@admin_dashboard_bp.route('/dashboard', methods=['GET'])
def admin_dashboard():
    if 'admin_id' not in session:
        print("Admin not logged in.")  # Debugging statement
        return "Access denied. Please log in as an admin.", 403

    print(f"Admin ID in session: {session['admin_id']}")  # Debugging statement

    conn = get_db_connection()
    students = conn.execute("SELECT * FROM students").fetchall()
    teachers = conn.execute("SELECT * FROM teachers").fetchall()
    conn.close()

    print("Students fetched:", students)  # Debugging statement
    print("Teachers fetched:", teachers)  # Debugging statement

    # Debugging statement to check if data is being passed to the template correctly
    print("Students being passed to the template:", students)
    print("Teachers being passed to the template:", teachers)

    return render_template('admin_dashboard.html', students=students, teachers=teachers)

# Delete Student Route
@admin_dashboard_bp.route('/delete_student/<int:id>', methods=['POST'])
def delete_student(id):
    conn = get_db_connection()
    
    # Delete the student record from the database
    conn.execute("DELETE FROM students WHERE Id = ?", (id,))
    conn.commit()

    # Delete the student's dataset images
    dataset_folder = 'dataset'  # Ensure this is the correct path to your dataset folder
    print("Files in dataset folder before deletion:", os.listdir(dataset_folder))  # Debugging line
    files_deleted = 0  # Counter for deleted files
    # Format the ID to always have two digits
    formatted_id = f"{id:02d}"  # This will format the ID to two digits (e.g., 01, 02, ..., 10, 11, ...)

    for file in os.listdir(dataset_folder):
        # Check if the filename starts with the formatted ID followed by an underscore
        if file.startswith(f"{formatted_id}_"):  # Match the format with leading zero
            file_path = os.path.join(dataset_folder, file)
            try:
                os.remove(file_path)  # Attempt to delete the file
                print(f"Successfully deleted: {file_path}")  # Debugging line
                files_deleted += 1  # Increment counter
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")  # Error handling

    if files_deleted == 0:
        print(f"No files found to delete for student ID: {id}")  # Debugging line

    conn.close()
    flash('Student deleted successfully!')
    return redirect(url_for('admin_dashboard.admin_dashboard'))
       
# Add Teacher and Subject Route
@admin_dashboard_bp.route('/add_teacher', methods=['POST'])
def add_teacher():
    teacher_name = request.form['teacher-name']
    subject_name = request.form['subject-name']

    conn = get_db_connection()
    conn.execute("INSERT INTO teachers (Name, Subject) VALUES (?, ?)", (teacher_name, subject_name))
    conn.commit()
    conn.close()

    flash('Teacher added successfully!')
    return redirect(url_for('admin_dashboard.admin_dashboard'))

# Delete Teacher Route
@admin_dashboard_bp.route('/delete_teacher/<int:id>', methods=['POST'])
def delete_teacher(id):
    conn = get_db_connection()
    conn.execute("DELETE FROM teachers WHERE id = ?", (id,))
    conn.commit()
    conn.close()

    flash('Teacher deleted successfully!')
    return redirect(url_for('admin_dashboard.admin_dashboard'))

# Logout Route
@admin_dashboard_bp.route('/logout')
def logout():
    session.clear()  # Clear the session
    return redirect(url_for('features.features'))

