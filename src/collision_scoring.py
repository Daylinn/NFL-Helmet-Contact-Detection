import pandas as pd
import os


proximity_path = "outputs/proximity/proximity_events.csv"
motion_path = "outputs/motion/motion_events.csv"

output_dir = "outputs/collisions/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "collision_likelihood.csv")


proximity_df = pd.read_csv(proximity_path)
motion_df = pd.read_csv(motion_path)


merged = proximity_df.merge(
    motion_df,
    on=["frame_id", "helmetA_index", "helmetB_index", "IoU"],
    how="inner"
)


merged["collision_score"] = merged["IoU"] * merged["max_pair_flow"]


merged = merged.sort_values(by="collision_score", ascending=False)


merged.to_csv(output_path, index=False)

print("Collision likelihood scoring complete!")
print("Saved to:", output_path)