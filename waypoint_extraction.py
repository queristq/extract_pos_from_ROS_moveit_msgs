import pickle
import numpy as np
import csv

# Load the pickle file
with open("dropoff_plan.pkl", "rb") as file:
    data = pickle.load(file)

# Save raw data to an output file for debugging (optional)
with open("output.txt", "w") as f:
    f.write(str(data))

def get_points(joint_traj):
    """Return the list of points from the joint trajectory object."""
    if isinstance(joint_traj, dict):
        return joint_traj.get("points", [])
    elif hasattr(joint_traj, "points"):
        return joint_traj.points
    return []

def get_positions(point):
    """Return the positions from a point."""
    if isinstance(point, dict):
        return point.get("positions")
    elif hasattr(point, "positions"):
        return point.positions
    return None

def extract_positions(item):
    """
    Extract positions from an item (which can be a dict or an object) 
    that holds trajectory data.
    """
    positions = []
    # Dictionary-based extraction
    if isinstance(item, dict):
        joint_traj = item.get("joint_trajectory")
        if joint_traj:
            pts = get_points(joint_traj)
            for pt in pts:
                pos = get_positions(pt)
                if pos is not None:
                    positions.append(list(pos))
    # Fallback: object attribute extraction
    elif hasattr(item, "joint_trajectory") and hasattr(item.joint_trajectory, "points"):
        pts = item.joint_trajectory.points
        for pt in pts:
            pos = get_positions(pt)
            if pos is not None:
                positions.append(list(pos))
    return positions

positions_data = []

# If the top-level object is a list, iterate over every item.
if isinstance(data, list):
    for item in data:
        positions_data.extend(extract_positions(item))
else:
    positions_data.extend(extract_positions(data))

# Check if we extracted any positions and then save to CSV.
if positions_data:
    trajectory_positions_array = np.array(positions_data)
    csv_file_path = "waypoints.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write column headers (assuming each waypoint has the same number of joints)
        writer.writerow([f"joint_{i+1}" for i in range(trajectory_positions_array.shape[1])])
        writer.writerows(trajectory_positions_array)
    print(f"Trajectory positions saved to {csv_file_path}")
else:
    print("Error: No positions data found in the trajectory.")
