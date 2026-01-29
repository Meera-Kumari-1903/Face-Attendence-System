# Face Authentication Attendance System Documentation

## 1. Model and Approach Used

This project implements a face authentication-based attendance system using:

- OpenCV for real-time face detection
- LBPH (Local Binary Pattern Histogram) Face Recognizer for identification
- Streamlit Dashboard for attendance log viewing

### Face Detection
Faces are detected using Haar Cascade Classifier:

- `haarcascade_frontalface_default.xml`

### Face Recognition Model
The recognition is performed using:

- LBPHFaceRecognizer (OpenCV Contrib)

LBPH is chosen because:

- Works well on small datasets
- Fast and lightweight
- Suitable for real-time applications
  

---

## 2. Training Process

### Step 1: Face Registration
A user registers their face using webcam input:

- Press `C` to capture face image
- Face image is stored in `faces/` folder

Example:
```bash
faces/Abhishek_1.jpg
```

### Step 2: Model Training
The training script reads all registered face images and assigns numeric labels.

Training is done using:

```python
recognizer.train(faces, np.array(labels))
```
The trained model is saved as:
```bash
face_model.yml
```
## 3. Accuracy Expectations

Since LBPH is a classical face recognition approach:

 - Accuracy is good for small groups (5–20 users)

 - Works best with proper lighting and frontal face images

Expected performance:

 - 80–90% accuracy in controlled indoor conditions

 - Lower accuracy in poor lighting or large datasets

Accuracy can be improved by:

 - Capturing multiple images per user

 - Using deep learning models like FaceNet

## 4. Known Failure Cases

This system may fail in the following situations:

 - Poor lighting conditions

 - Multiple faces present in camera frame

 - Face partially covered (mask, hand, scarf)

 - Extreme head tilt or side angles

 - Low resolution webcam quality

## 5. Spoof Prevention

A basic constraint is applied:

 - Only one face allowed at a time

Advanced spoof detection (blink/liveness) is not fully implemented due to time constraints.

## 6. Deliverables Provided

- Complete Source Code in GitHub Repository

- Working Demo Video showing:

 - User Registration

 - Training Process

 - Punch-In / Punch-Out Attendance

 - Dashboard Logs

- Attendance stored in CSV format

## 7. How to Run the Project
Install Requirements:
```bash
pip install -r requirements.txt
```
Register User
```bash
python register_face.py
```
Train Model
```bash
python train_model.py
```
Mark Attendance
```bash
python attendance.py
```
View Dashboard
```bash
streamlit run dashboard.py
```

Author

Meera Kumari
AI/ML Internship Assignment Submission
