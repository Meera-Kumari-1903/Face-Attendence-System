# 🎯 Face Authentication Attendance System (OpenCV + Streamlit)

This project is a Face Recognition based Attendance System built using:

- OpenCV (Face Detection + Recognition)
- LBPH Face Recognizer
- Streamlit Dashboard for Attendance Logs

---

## ✅ Features

- Register a user face
- Train face recognition model
- Punch-In / Punch-Out Attendance
- Attendance stored in CSV file
- Dashboard to view logs and download report

---

## ⚠ Important Note

This project uses OpenCV webcam access:

```python
cv2.VideoCapture(0)
cv2.imshow()
```
So the camera will open as a desktop popup window, not inside the browser.

This project is meant to run locally on a laptop/PC.

## 📂 Project Structure

```
FaceAttendanceSystem/
│
├── register_face.py
├── train_model.py
├── attendance.py
├── dashboard.py
├── requirements.txt
├── DOCUMENTATION.md
└── Face_Attendance_Report.pdf
```
## 🔧 Installation
Step 1: Clone Repository

Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶ Usage Instructions

## 1️⃣ Register User Face

Run:
```
python register_face.py
```
Controls:

Press C → Capture Face Image

Press ESC → Exit

Captured image is saved inside:
```
faces/
```
## 2️⃣ Train Face Recognition Model

Run:
```bash
python train_model.py
```
This generates:
```bash
face_model.yml
```
## 3️⃣ Mark Attendance (Punch In/Out)

Run:
```bash
python attendance.py
```
## 4️⃣ View Attendance Dashboard

Run:
```bash
streamlit run dashboard.py
```
## 📌 Output Files

faces/ → Registered face images

face_model.yml → Trained model

attendance.csv → Attendance log
