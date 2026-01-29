# ==============================================
# 🎯 Face Authentication Attendance Dashboard
# Author: Abhishek Project
# Technology: Streamlit + OpenCV + LBPH
# Features:
#   ✅ Register User Face
#   ✅ Auto Train Model After Capture
#   ✅ Punch-In / Punch-Out Attendance
#   ✅ OS Camera Popup with Key Controls
#   ✅ View Attendance Logs in Dashboard
# ==============================================


import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from datetime import datetime


# ==============================================
# STREAMLIT PAGE SETTINGS
# ==============================================

st.set_page_config(page_title="Face Attendance System", layout="wide")

st.title("🎯 Face Authentication Attendance Dashboard")


# ==============================================
# FACE DETECTOR (Haar Cascade)
# ==============================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ==============================================
# FILE & FOLDER SETUP
# ==============================================

if not os.path.exists("faces"):
    os.makedirs("faces")

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv("attendance.csv", index=False)


# ==============================================
# TRAIN MODEL FUNCTION
# ==============================================

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for file in os.listdir("faces"):
        if file.endswith(".jpg"):
            name = file.split("_")[0]

            if name not in label_map:
                label_map[name] = current_id
                current_id += 1

            img_path = os.path.join("faces", file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            faces.append(img)
            labels.append(label_map[name])

    if len(faces) == 0:
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save("face_model.yml")

    print("✅ Model Trained Successfully!")
    return True


# ==============================================
# REGISTER USER FUNCTION (Single Capture + Auto Close)
# ==============================================

def register_user(name):
    cam = cv2.VideoCapture(0)

    print("\n📌 Registration Mode")
    print("Press C to Capture Face (Camera closes immediately)")
    print("Press ESC to Exit\n")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, "Press C to Capture | ESC to Exit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        cv2.imshow("Register User Camera", frame)

        key = cv2.waitKey(1)

        # ✅ Capture Face and Close Immediately
        if key == ord("c"):

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y+h, x:x+w]

                file_path = f"faces/{name}_1.jpg"
                cv2.imwrite(file_path, face_img)

                print("✅ Face Captured Successfully!")
                break

            elif len(faces) > 1:
                print("❌ Multiple faces detected! Only one allowed.")

            else:
                print("❌ No face detected!")

        # ✅ ESC Exit
        elif key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    # ✅ Auto Train After Registration
    print("\n🧠 Training Model Automatically...")
    train_model()
    print("✅ Model Trained Successfully!")


# ==============================================
# ATTENDANCE FUNCTION (Single Confirm + Auto Close)
# ==============================================

def punch_attendance(status):

    if not os.path.exists("face_model.yml"):
        st.error("❌ Model not trained yet! Register first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    # Load Registered Users
    users = {}
    id_counter = 0

    for file in os.listdir("faces"):
        if file.endswith(".jpg"):
            name = file.split("_")[0]
            if name not in users.values():
                users[id_counter] = name
                id_counter += 1

    cam = cv2.VideoCapture(0)

    print(f"\n📌 {status} Mode")
    print("Press C to Confirm Attendance (Camera closes immediately)")
    print("Press ESC to Exit\n")

    detected_name = None

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]

            label, confidence = recognizer.predict(face_img)

            if confidence < 70:
                detected_name = users[label]

                cv2.putText(frame, detected_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Press C to Confirm | ESC to Exit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow("Attendance Camera", frame)

        key = cv2.waitKey(1)

        # ✅ Confirm Attendance + Close Camera
        if key == ord("c"):

            if detected_name:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                df = pd.read_csv("attendance.csv")
                df.loc[len(df)] = [detected_name, now, status]
                df.to_csv("attendance.csv", index=False)

                print(f"✅ {detected_name} Marked {status}")
                break
            else:
                print("❌ Face not recognized!")

        # ✅ ESC Exit
        elif key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


# ==============================================
# STREAMLIT DASHBOARD UI
# ==============================================

st.sidebar.header("⚙ Control Panel")

option = st.sidebar.radio(
    "Select Action",
    ["Register User", "Punch In", "Punch Out", "View Logs"]
)


# ==============================================
# REGISTER USER UI
# ==============================================

if option == "Register User":
    st.subheader("👤 Register New User")

    name = st.text_input("Enter User Name")

    st.info("📌 Camera Keys: Press C to Capture Face | ESC to Exit")

    if st.button("📸 Register User (Auto Train)"):
        if name.strip() == "":
            st.warning("Enter a valid name!")
        else:
            register_user(name)
            st.success("✅ User Registered + Model Trained Successfully!")


# ==============================================
# PUNCH IN UI
# ==============================================

elif option == "Punch In":
    st.subheader("✅ Punch In Attendance")

    st.info("📌 Camera Keys: Press C to Confirm Punch-In | ESC to Exit")

    if st.button("Open Camera for Punch-In"):
        punch_attendance("Punch-In")
        st.success("✅ Punch-In Updated Successfully!")


# ==============================================
# PUNCH OUT UI
# ==============================================

elif option == "Punch Out":
    st.subheader("🚪 Punch Out Attendance")

    st.info("📌 Camera Keys: Press C to Confirm Punch-Out | ESC to Exit")

    if st.button("Open Camera for Punch-Out"):
        punch_attendance("Punch-Out")
        st.success("✅ Punch-Out Updated Successfully!")


# ==============================================
# VIEW LOGS UI
# ==============================================

elif option == "View Logs":
    st.subheader("📊 Attendance Logs")

    df = pd.read_csv("attendance.csv")
    st.dataframe(df, width="stretch")
