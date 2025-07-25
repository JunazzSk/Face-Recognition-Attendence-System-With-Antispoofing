from flask import Blueprint, request, jsonify, session
import sqlite3

admin_auth_bp = Blueprint('admin_auth', __name__)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('sqlite.db')
    conn.row_factory = sqlite3.Row
    return conn

# Admin Login Route
@admin_auth_bp.route('/login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('adminId')
    password = data.get('adminPassword')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE AdminID = ? AND Password = ?", (username, password))
    admin = cursor.fetchone()
    conn.close()

    if admin:
        session['admin_id'] = admin['AdminID']
        return jsonify({"message": "Login successful", "redirect_url": "/features/admin_dashboard"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# Admin Signup Route
@admin_auth_bp.route('/signup', methods=['POST'])
def admin_signup():
    data = request.json
    new_admin_id = data.get('newAdminId')
    new_admin_password = data.get('newAdminPassword')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO admins (AdminID, Password) VALUES (?, ?)", (new_admin_id, new_admin_password))
    conn.commit()
    conn.close()

    return jsonify({"message": "Signup successful"}), 200