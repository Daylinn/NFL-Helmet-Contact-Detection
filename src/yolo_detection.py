from ultralytics import YOLO
import cv2
import csv
import os

# Load YOLO model
model = YOLO("yolov8s.pt")

# Path to ONE test video
video_path = "data/raw/test/57906_000718_Endzone.mp4"

# Output folder
output_dir = "outputs/detections/"
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_id = 0

# CSV for detections
csv_path = os.path.join(output_dir, "helmet_detections.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "confidence", "class"])

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            writer.writerow([frame_id, x1, y1, x2, y2, conf, cls])

    frame_id += 1

csv_file.close()
cap.release()

print("Detection complete! CSV saved to:", csv_path)