import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

print("\n📌 Training Model...\n")

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
    print("❌ No registered faces found. Run register_face.py first.")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

print("✅ Model Trained Successfully!")
print("Registered Users:", label_map)
