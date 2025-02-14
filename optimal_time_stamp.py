import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

# =====================================
# 1. Load Waypoints Data from CSV File
# =====================================
# Make sure "waypoints.csv" is in your working directory.
df = pd.read_csv("dropoff_plan_waypoints.csv")
q = df.values  # Shape: (N, 6) for a 6-DOF robot.
N, num_joints = q.shape

# =====================================
# 2. Define Joint Acceleration Limits
# =====================================
# Example acceleration limits (adjust these to match your robot).
acc_limits = np.array([10.0, 1.9, 2.0, 15.0, 14.0, 17.0])  # [rad/s²]

# ======================================================
# 3. Define the Optimization Problem and Helper Functions
# ======================================================
# Decision variables: the time intervals (dt) between waypoints.
# Total trajectory time T = sum(dt).

def total_time(dt):
    """Objective: Minimize the total trajectory time."""
    return np.sum(dt)

def acceleration_violations(dt):
    """
    For each interior waypoint (i = 1, ..., N-2) and for each joint j,
    compute the finite-difference acceleration:
        v_prev = (q[i, j] - q[i-1, j]) / dt[i-1]
        v_next = (q[i+1, j] - q[i, j]) / dt[i]
        a_i    = (v_next - v_prev) / (0.5 * (dt[i-1] + dt[i]))
    Returns |a_i| - acc_limits[j] (must be <= 0).
    """
    dt = np.asarray(dt)
    violations = []
    for i in range(1, N - 1):
        for j in range(num_joints):
            v_prev = (q[i, j] - q[i - 1, j]) / dt[i - 1]
            v_next = (q[i + 1, j] - q[i, j]) / dt[i]
            avg_dt = 0.5 * (dt[i - 1] + dt[i])
            a = (v_next - v_prev) / avg_dt
            violations.append(np.abs(a) - acc_limits[j])
    return np.array(violations)

# Nonlinear constraint: all acceleration violations must be <= 0.
nlc = NonlinearConstraint(acceleration_violations, -np.inf, 0)

# ================================================
# 4. Set Initial Guess and Bounds for the Optimization
# ================================================
# Assume an initial total trajectory time (e.g., 5 seconds) with equal dt.
initial_total_time = 5.0
initial_dt = np.ones(N - 1) * (initial_total_time / (N - 1))
bounds = [(0.01, None) for _ in range(N - 1)]

# ======================================
# 5. Run the Optimization (SLSQP)
# ======================================
res = minimize(
    total_time,
    initial_dt,
    method='SLSQP',
    bounds=bounds,
    constraints=[nlc],
    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True}
)

if res.success:
    opt_dt = res.x
    # Construct optimized time stamps:
    t_opt = np.concatenate(([0], np.cumsum(opt_dt)))
    print("Optimized total time:", t_opt[-1])
else:
    raise RuntimeError("Optimization failed: " + res.message)

# ======================================
# 6. Compute Velocities and Accelerations
# ======================================
def compute_vel_and_acc(t, q):
    """
    Compute velocities and accelerations using finite differences.
    For velocities:
      - Use forward difference at the first point,
      - Backward difference at the last point,
      - Central difference for interior points.
    Similarly for accelerations.
    """
    N = len(t)
    vel = np.zeros_like(q)
    acc = np.zeros_like(q)
    for i in range(N):
        if i == 0:
            dt_forward = t[1] - t[0]
            vel[i] = (q[1] - q[0]) / dt_forward
        elif i == N - 1:
            dt_backward = t[-1] - t[-2]
            vel[i] = (q[-1] - q[-2]) / dt_backward
        else:
            dt_central = t[i + 1] - t[i - 1]
            vel[i] = (q[i + 1] - q[i - 1]) / dt_central
    for i in range(N):
        if i == 0:
            dt_forward = t[1] - t[0]
            acc[i] = (vel[1] - vel[0]) / dt_forward
        elif i == N - 1:
            dt_backward = t[-1] - t[-2]
            acc[i] = (vel[-1] - vel[-2]) / dt_backward
        else:
            dt_central = t[i + 1] - t[i - 1]
            acc[i] = (vel[i + 1] - vel[i - 1]) / dt_central
    return vel, acc

# Compute velocities and accelerations for optimized time stamps.
vel_opt, acc_opt = compute_vel_and_acc(t_opt, q)

# For comparison, compute baseline (equal spacing) timing:
t_equal = np.linspace(0, initial_total_time, N)
vel_equal, acc_equal = compute_vel_and_acc(t_equal, q)

# ======================================
# 7. Generate the CSV File with Time, Positions, Velocities, and Accelerations
# ======================================
header = (["time"] +
          [f"pos_joint_{i+1}" for i in range(num_joints)] +
          [f"vel_joint_{i+1}" for i in range(num_joints)] +
          [f"acc_joint_{i+1}" for i in range(num_joints)])

data_out = np.column_stack((t_opt, q, vel_opt, acc_opt))
trajectory_df = pd.DataFrame(data_out, columns=header)
output_filename = "optimized_joint_trajectory_with_vel_acc.csv"
trajectory_df.to_csv(output_filename, index=False)
print(f"CSV file '{output_filename}' has been generated.")

# ======================================
# 8. Visualization: Save Comparison Figures
# ======================================

# -- Position Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, q[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, q[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Pos')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Position Comparison", y=1.02)
plt.tight_layout()
plt.savefig("position_comparison.png")
print("Position comparison figure saved as 'position_comparison.png'")

# -- Velocity Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, vel_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, vel_opt[:, j], 's-', label='Optimized')
    axs[j].set_ylabel(f'Joint {j+1} Vel')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Velocity Comparison", y=1.02)
plt.tight_layout()
plt.savefig("velocity_comparison.png")
print("Velocity comparison figure saved as 'velocity_comparison.png'")

# -- Acceleration Comparison --
fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
for j in range(num_joints):
    axs[j].plot(t_equal, acc_equal[:, j], 'o-', label='Equal spacing')
    axs[j].plot(t_opt, acc_opt[:, j], 's-', label='Optimized')
    # Plot acceleration limits for reference.
    axs[j].axhline(acc_limits[j], color='r', linestyle='--', label='Acc limit' if j == 0 else "")
    axs[j].axhline(-acc_limits[j], color='r', linestyle='--')
    axs[j].set_ylabel(f'Joint {j+1} Acc')
    axs[j].legend(loc='upper right')
axs[-1].set_xlabel('Time [s]')
fig.suptitle("Joint Acceleration Comparison", y=1.02)
plt.tight_layout()
plt.savefig("acceleration_comparison.png")
print("Acceleration comparison figure saved as 'acceleration_comparison.png'")
