// Navigation Functionality
function navigateTo(page) {
  window.location.href = page;
}

// Countdown for Form Submission
document.getElementById('dataForm').addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent the default form submission

  // Show countdown message
  document.getElementById('countdownMessage').style.display = 'block';
  let countdown = 3;
  document.getElementById('countdown').innerText = countdown;

  const countdownInterval = setInterval(function() {
      countdown--;
      document.getElementById('countdown').innerText = countdown;

      if (countdown <= 0) {
          clearInterval(countdownInterval);
          document.getElementById('dataForm').submit(); // Submit the form after countdown
      }
  }, 1000);
});

// Modal Management
function showModal(title) {
  document.getElementById("modal-title").innerText = title;
  document.getElementById("adminModal").style.display = "block";
}

function closeModal() {
  document.getElementById("adminModal").style.display = "none";
  document.getElementById("adminForm").reset();
}

// Admin Form Submission
function submitAdminForm() {
  const newAdminId = document.getElementById("newAdminId").value;
  const newAdminPassword = document.getElementById("newAdminPassword").value;

  fetch("/admin_auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ newAdminId, newAdminPassword }),
  })
  .then((response) => {
      if (response.ok) {
          alert("Signup successful!");
          closeModal();
      } else {
          return response.json().then((data) => alert(data.error));
      }
  })
  .catch((error) => console.log("Error during signup:", error));
}

// Admin Login Handling
function handleLogin(event) {
    event.preventDefault();
    const adminId = document.getElementById("adminId").value;
    const adminPassword = document.getElementById("adminPassword").value;
  
    fetch("/admin_auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ adminId, adminPassword }),
    })
    .then((response) => {
        if (response.ok) {
            return response.json();
        } else {
            return response.json().then((data) => {
                throw new Error(data.error);
            });
        }
    })
    .then((data) => {
        alert(data.message);
        window.location.href = "/admin/dashboard"; // Update the URL to match the one defined in admin_dashboard.py
    })
    .catch((error) => alert(error.message));
  }

// Data Training Functionality
function trainData() {
  fetch('/features/trainData', {  // Ensure this URL is correct
      method: 'POST'
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok ' + response.statusText);
      }
      return response.json();
  })
  .then(data => {
      alert(data.message);
  })
  .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while training the data: ' + error.message);
  });
}

// Display Selected Values on Attendance Page
if (window.location.pathname.includes('/attendance')) {
  const params = getQueryParams();
  const attendanceInfo = document.getElementById('attendanceInfo');
  attendanceInfo.innerText = `Teacher: ${params.teacher}, Subject: ${params.subject}, Period: ${params.period}`;
}

function downloadAttendance() {
    // Get the selected filter values
    const date = document.getElementById('date').value;
    const teacher = document.getElementById('teacher').value;
    const subject = document.getElementById('subject').value;

    // Create a form dynamically
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = '/features/download_attendance'; // Use the correct Flask route
    document.body.appendChild(form);

    // Add hidden inputs for the filter values
    const dateInput = document.createElement('input');
    dateInput.type = 'hidden';
    dateInput.name = 'date';
    dateInput.value = date;
    form.appendChild(dateInput);

    const teacherInput = document.createElement('input');
    teacherInput.type = 'hidden';
    teacherInput.name = 'teacher';
    teacherInput.value = teacher;
    form.appendChild(teacherInput);

    const subjectInput = document.createElement('input');
    subjectInput.type = 'hidden';
    subjectInput.name = 'subject';
    subjectInput.value = subject;
    form.appendChild(subjectInput);

    // Submit the form
    form.submit();
}
function deleteAttendance(studentId, date, time, teacher, subject, period) {
    if (confirm("Are you sure you want to delete this attendance record?")) {
        fetch('/features/delete_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_id: studentId,
                date: date,
                time: time,
                teacher: teacher,
                subject: subject,
                period: period
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || "Failed to delete the attendance record.");
                });
            }
            return response.json();
        })
        .then(data => {
            showSuccessMessage(data.message); // Show custom success message
            setTimeout(() => {
                window.location.reload(); // Refresh the page after 2 seconds
            }, 2000);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(error.message); // Show specific error message
        });
    }
}

function showSuccessMessage(message) {
    const popup = document.getElementById('success-message');
    if (!popup) {
        console.error("Error: 'success-message' element not found in the DOM.");
        return;
    }
    popup.textContent = message; // Set the message text
    popup.style.display = 'block'; // Show the popup

    // Hide the popup after 2 seconds
    setTimeout(() => {
        popup.style.display = 'none';
    }, 2000);
}