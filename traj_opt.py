import pickle
import numpy as np
import csv

# Load the pickle file
with open("pickup_plan.pkl", "rb") as file:
    data = pickle.load(file)

# Save the raw data to an output file for debugging (optional)
with open("output.txt", "w") as f:
    print(data, file=f)

# Extract positions from the RobotTrajectory object correctly
positions_data = []
if hasattr(data, "joint_trajectory") and hasattr(data.joint_trajectory, "points"):
    for point in data.joint_trajectory.points:
        positions_data.append(list(point.positions))  # Ensure it's stored as a list

# Ensure data is not empty
if positions_data:
    # Convert to a NumPy array
    trajectory_positions_array = np.array(positions_data)

    # Save to CSV file
    csv_file_path = "trajectory_positions.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"joint_{i+1}" for i in range(trajectory_positions_array.shape[1])])  # Column headers
        writer.writerows(trajectory_positions_array)  # Data rows

    print(f"Trajectory positions saved to {csv_file_path}")
else:
    print("Error: No positions data found in the trajectory.")
