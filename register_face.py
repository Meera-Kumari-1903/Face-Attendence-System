import cv2
import os

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = input("Enter User Name: ")

if not os.path.exists("faces"):
    os.makedirs("faces")

cam = cv2.VideoCapture(0)

print("\n📌 Registration Mode")
print("Press C to Capture Face Image (Camera closes immediately)")
print("Press ESC to Exit\n")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    cv2.putText(frame, "Press C to Capture | ESC to Exit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1)

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

    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()
