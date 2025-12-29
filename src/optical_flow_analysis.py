import os
import cv2
import numpy as np
import pandas as pd


video_path = "data/raw/test/57906_000718_Endzone.mp4"  # <- set to the SAME video used for YOLO
detections_path = "outputs/detections/helmet_detections.csv"
proximity_path = "outputs/proximity/proximity_events.csv"

output_dir = "outputs/motion/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "motion_events.csv")


detections_df = pd.read_csv(detections_path)
proximity_df = pd.read_csv(proximity_path)


detections_df = detections_df.reset_index()  
detections_df.rename(columns={"index": "det_index"}, inplace=True)


frame_groups = detections_df.groupby("frame_id")


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

prev_gray = None
frame_id = -1


max_flow_by_det_index = {}

# Farnebäck parameters (standard starting point)
farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

print("Computing optical flow and max motion per helmet box...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue  

    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_params)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

    
    if frame_id in frame_groups.groups:
        frame_dets = frame_groups.get_group(frame_id)

        for _, row in frame_dets.iterrows():
            det_idx = int(row["det_index"])
            x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]].astype(int)

            # Clamp to frame bounds
            h, w = gray.shape
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                continue  # invalid bbox

            region_mag = mag[y1:y2, x1:x2]
            if region_mag.size == 0:
                continue

            max_mag = float(region_mag.max())
            max_flow_by_det_index[det_idx] = max_mag

    prev_gray = gray

cap.release()
print("Finished computing max flow per detection.")


motion_records = []

for _, row in proximity_df.iterrows():
    frame_id = int(row["frame_id"])
    idx_a = int(row["helmetA_index"])
    idx_b = int(row["helmetB_index"])
    iou = float(row["IoU"])

    max_flow_a = max_flow_by_det_index.get(idx_a, np.nan)
    max_flow_b = max_flow_by_det_index.get(idx_b, np.nan)

    
    if np.isnan(max_flow_a) or np.isnan(max_flow_b):
        continue

    max_pair_flow = max(max_flow_a, max_flow_b)

    motion_records.append({
        "frame_id": frame_id,
        "helmetA_index": idx_a,
        "helmetB_index": idx_b,
        "IoU": iou,
        "max_flow_A": max_flow_a,
        "max_flow_B": max_flow_b,
        "max_pair_flow": max_pair_flow
    })

motion_df = pd.DataFrame(motion_records)
motion_df.to_csv(output_path, index=False)

print("Motion analysis complete! Saved to:", output_path)