import pickle
import pandas as pd
import json

# Load the optimized joint trajectory data from CSV
csv_file = "final_trajectory_bspline.csv"
df = pd.read_csv(csv_file)

# Extract joint names (assuming they follow the format 'pos_joint_x')
joint_names = [col.replace("pos_", "") for col in df.columns if "pos_joint" in col]

# Construct the MoveIt-style trajectory message
moveit_msg = {
    "joint_trajectory": {
        "header": {
            "seq": 0,
            "stamp": {
                "secs": 0,
                "nsecs": 0
            },
            "frame_id": "base_link"
        },
        "joint_names": joint_names,
        "points": []
    },
    "multi_dof_joint_trajectory": {
        "header": {
            "seq": 0,
            "stamp": {
                "secs": 0,
                "nsecs": 0
            },
            "frame_id": ""
        },
        "joint_names": [],
        "points": []
    }
}

# Extract waypoints and construct points list
for i, row in df.iterrows():
    point = {
        "positions": row[[col for col in df.columns if "pos_joint" in col]].tolist(),
        "velocities": row[[col for col in df.columns if "vel_joint" in col]].tolist(),
        "accelerations": row[[col for col in df.columns if "acc_joint" in col]].tolist(),
        "effort": [],
        "time_from_start": {
            "secs": int(row["time"]),
            "nsecs": int((row["time"] - int(row["time"])) * 1e9)
        }
    }
    moveit_msg["joint_trajectory"]["points"].append(point)

# Save MoveIt message to a text file for validation
validation_text_file = "moveit_trajectory.txt"
with open(validation_text_file, "w") as file:
    file.write(json.dumps(moveit_msg, indent=2))

print(f"MoveIt trajectory message saved to {validation_text_file} for validation.")

# Save the MoveIt message using pickle
output_pickle_file = "moveit_trajectory.pkl"
with open(output_pickle_file, "wb") as file:
    pickle.dump(moveit_msg, file)

print(f"MoveIt trajectory message saved to {output_pickle_file}")
