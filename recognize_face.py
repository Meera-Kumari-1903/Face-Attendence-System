import cv2
import pandas as pd
import os
from datetime import datetime

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

users = {}
id_counter = 0

for file in os.listdir("faces"):
    if file.endswith(".jpg"):
        name = file.split("_")[0]
        if name not in users.values():
            users[id_counter] = name
            id_counter += 1

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv("attendance.csv", index=False)

cam = cv2.VideoCapture(0)

print("\n📌 Attendance Mode")
print("Press I for Punch-In")
print("Press O for Punch-Out")
print("Press ESC to Exit\n")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    detected_name = None

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_img)

        if confidence < 70:
            detected_name = users[label]

            cv2.putText(frame, detected_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (255, 0, 0), 2)

    cv2.putText(frame, "I=PunchIn | O=PunchOut | ESC=Exit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1)

    if key == ord("i") and detected_name:
        status = "Punch-In"

    elif key == ord("o") and detected_name:
        status = "Punch-Out"

    else:
        status = None

    if status:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.read_csv("attendance.csv")
        df.loc[len(df)] = [detected_name, now, status]
        df.to_csv("attendance.csv", index=False)

        print(f"✅ {detected_name} Marked {status}")

        break

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
