import pandas as pd
import os
from itertools import combinations

def compute_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Determine intersections
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Areas of each box
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - intersection

    if union == 0:
        return 0
    return intersection / union


detections_path = "outputs/detections/helmet_detections.csv"
df = pd.read_csv(detections_path)


frames = df.groupby("frame_id")


IOU_THRESHOLD = 0.10


proximity_events = []


for frame_id, group in frames:
    boxes = group[["x1", "y1", "x2", "y2"]].values
    indices = list(group.index)

    # Compare ALL helmet pairs in the frame
    for (idx_a, box_a), (idx_b, box_b) in combinations(zip(indices, boxes), 2):
        iou = compute_iou(box_a, box_b)

        if iou > IOU_THRESHOLD:
            proximity_events.append({
                "frame_id": frame_id,
                "helmetA_index": idx_a,
                "helmetB_index": idx_b,
                "IoU": iou
            })


output_dir = "outputs/proximity/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "proximity_events.csv")
pd.DataFrame(proximity_events).to_csv(output_path, index=False)

print("Proximity analysis complete! Saved to:", output_path)